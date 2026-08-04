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
session id: 019fc2f8-a371-7da0-8c7a-70a10de177bf
--------
user
Round-7 review: the new distance-to-modeled-limit analysis (last section of docs/performance/scaling/levante_campaign_2026-07-24.md) + instruments. Attack:
1. Census probe scripts/tmp/_probe_latlon_halo_census.py: virtual-CPU forced-host lowering of make_sharded_atm_latlon_step, counted 41 collective-permute + 1 all-reduce per step, identical at nd=8/16; exact CP result bytes summed from compiled HLO text via regex on 'collective-permute(' lines (4,635,408 B at n_lon=1024 L26 f32; linear in n_lon 0.07% residual). Is the regex robust (start/done pairs double-count? async pairs? result-shape vs operand-shape)? Is nd-independence at 8/16 sufficient to claim it for 128?
2. Bound arithmetic: calibrated_bound with compute=nd1 same-tile receipts (job 26630370: 1.659/2.837/2.973 ms), msgs=41, bytes=18.5MB (f32@4096) scaled x2 for f64, nred=1, imbalance from rows, IB 26.3us/23.5GB/s: t_bound 2.863/1.848/2.999 -> measured/bound 2.35/3.02/3.20 for LL2048@64 f32 / @128 f32 / @128 f64. Recheck numbers (measured 6.7324/5.5767/9.6015).
3. Claims wording: '~2.4-3.2x above modeled limit', mechanism labelled PLAUSIBLE (per-CP effective overhead, arXiv:2607.16100 regime), A/B lever job 26630438 with falsifiability block (CONFIRM >=10%, REFUTE <=2%). Any overclaim/confound (e.g. f64 bytes: do 2-D f32 fields stay f32 under x64? census was f32-only), pole-tile compute bias direction, the ocean-null distinction argument.
End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + numbered issues.
codex
I’ll audit the cited section and probe script against the available artifacts, then recalculate the bound from the stated inputs and flag any claims that need narrowing.
exec
/usr/bin/bash -lc "pwd && rg --files -g 'levante_campaign_2026-07-24.md' -g '_probe_latlon_halo_census.py' -g '26630370*' -g '26630438*' -g '*.csv' -g '*.json' | sed -n '1,240p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 434ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.zenodo.json
tests/visual_baselines/sw_cube_vwind.json
docs/performance/scaling/barotropic_mcut.csv
docs/performance/scaling/barotropic_mcut_np4.csv
docs/performance/scaling/barotropic_mcut_np2.csv
docs/performance/scaling/scaling_indicators.csv
docs/performance/scaling/levante_campaign_2026-07-24.md
tests/ocean/fidelity/fixtures/tier8_global_realistic.json
tests/ocean/fidelity/fixtures/tier5_baroclinic_instability.json
tests/ocean/fidelity/fixtures/tier0_invariants.json
tests/ocean/fidelity/fixtures/tier6_channel_circulation.json
tests/ocean/fidelity/fixtures/tier1_linear_waves.json
tests/ocean/fidelity/fixtures/tier7_dino.json
tests/ocean/fidelity/fixtures/tier4_wind_driven_gyres.json
tests/ocean/fidelity/fixtures/tier3_process_benchmarks.json
tests/ocean/fidelity/fixtures/tier2_geostrophic_thermalwind.json
tests/unit/baselines/aerobulk_noskin_v1.json
tests/validation/scm_oracle/manifest.json
tests/unit/golden/yaml_to_experiment/williamson_test2.json
tests/unit/golden/yaml_to_experiment/williamson_test5.json
config/wb/sota/wb2_headline_rmse.csv
config/wb/sweep/stage1/manifest.json
scripts/cluster/ec_site/ec_site_soil_texture.csv
config/aimip/t106_headtohead/winner.json
scripts/validate/ec_site_regression_lai.csv
config/aimip/sweep/stage2/stage1_winners.json
config/aimip/sweep/stage2/manifest.json
scripts/validate/ec_site_regression_golden.json
config/aimip/sweep/stage1/manifest.json

exec
/usr/bin/bash -lc "rg -n -C 5 \"modeled limit|calibrated_bound|26630370|26630438|distance-to|collective-permute|ocean-null|PL[A-Z]+IBLE|CONFIRM|REFUTE\" docs/performance/scaling/levante_campaign_2026-07-24.md scripts tmp . --glob '"'!*.ipynb'"' 2>/dev/null" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 2 in 4207ms:
docs/performance/scaling/levante_campaign_2026-07-24.md-100-OPEN ANOMALY, characterised not explained: per-GPU throughput dips at
docs/performance/scaling/levante_campaign_2026-07-24.md-101-np=4 (247 Mc/s/GPU vs 430 at np2 and 306 at np8), so np4 is barely faster
docs/performance/scaling/levante_campaign_2026-07-24.md-102-than np2 while np8 is 2.5x faster than np4. Evidence gathered:
docs/performance/scaling/levante_campaign_2026-07-24.md-103-- REPRODUCIBLE: two repeats per arm agree within 1 % (19.92/19.87,
docs/performance/scaling/levante_campaign_2026-07-24.md-104-  17.14/17.04, 6.87/6.97).
docs/performance/scaling/levante_campaign_2026-07-24.md:105:- PLACEMENT REFUTED: np4 packed on one node (17.09 ms) == np4 spread over
docs/performance/scaling/levante_campaign_2026-07-24.md-106-  two nodes (17.12 ms), so node crossing is irrelevant.
docs/performance/scaling/levante_campaign_2026-07-24.md:107:- HALO VOLUME REFUTED: ghost-cell census on the padded mesh gives
docs/performance/scaling/levante_campaign_2026-07-24.md-108-  1540/1587/1400/1136 ghost cells per device at np 2/4/8/16 — flat to
docs/performance/scaling/levante_campaign_2026-07-24.md-109-  falling, and under 3 % of owned cells at every count.
docs/performance/scaling/levante_campaign_2026-07-24.md:110:- COLLECTIVE COUNT REFUTED (HLO census, ico L7, CPU virtual devices —
docs/performance/scaling/levante_campaign_2026-07-24.md-111-  device count is a compile-time property so the HLO matches what the GPUs
docs/performance/scaling/levante_campaign_2026-07-24.md:112:  execute): collective-permutes per step are 3 / 9 / 21 at np 2/4/8, i.e.
docs/performance/scaling/levante_campaign_2026-07-24.md-113-  np8 issues 2.3x MORE collectives than np4 and still runs 2.5x faster.
docs/performance/scaling/levante_campaign_2026-07-24.md-114-  Collective COUNT therefore cannot explain the np4 dip (this assumes cost
docs/performance/scaling/levante_campaign_2026-07-24.md-115-rises with count; a per-message-size effect is not excluded). (Fusion count 136/173/240,
docs/performance/scaling/levante_campaign_2026-07-24.md-116-  bitcasts 526/582/694 — the np8 program is finer-grained.)
docs/performance/scaling/levante_campaign_2026-07-24.md:117:- PARTITION METHOD REFUTED (job 26455829): the dip is method-independent —
docs/performance/scaling/levante_campaign_2026-07-24.md-118-  np4/np8 = 17.22/7.00 ms (sfc), 17.20/6.97 (metis), 19.35/6.26
docs/performance/scaling/levante_campaign_2026-07-24.md-119-  (geometric). Every method shows the same 2.5-3.1x jump.
docs/performance/scaling/levante_campaign_2026-07-24.md:120:- XLA CODEGEN ENV KNOBS REFUTED (job 26455948): np4 is 17.12 ms base,
docs/performance/scaling/levante_campaign_2026-07-24.md-121-  17.06 autotune-level-4, 17.10 latency-hiding-off, 17.01
docs/performance/scaling/levante_campaign_2026-07-24.md-122-  command-buffers-off — every arm within 1 %, none recovers np4.
docs/performance/scaling/levante_campaign_2026-07-24.md-123-  (The multi-output-fusion arm errored on an unsupported flag name and is
docs/performance/scaling/levante_campaign_2026-07-24.md-124-  not counted.)
docs/performance/scaling/levante_campaign_2026-07-24.md-125-VERDICT: five hypotheses refuted by measurement (placement, halo volume,
--
docs/performance/scaling/levante_campaign_2026-07-24.md-132-PRACTICAL GUIDANCE MEANWHILE: run this grid at np>=8, where per-device
docs/performance/scaling/levante_campaign_2026-07-24.md-133-throughput is 304-340 Mc/s/GPU vs 220-248 at np4.
docs/performance/scaling/levante_campaign_2026-07-24.md-134-
docs/performance/scaling/levante_campaign_2026-07-24.md-135-RESOLVED 2026-07-26 (nsys job 26479922 + HLO dump 26480096 + sqlite
docs/performance/scaling/levante_campaign_2026-07-24.md-136-timeline): the dip is an XLA CODEGEN pathology, localized to named
docs/performance/scaling/levante_campaign_2026-07-24.md:137:kernels. CONFIRMED: (1) the dip reproduces under nsys with matched
docs/performance/scaling/levante_campaign_2026-07-24.md-138-protocol (L8, padded-16 mesh: 21.01/17.78/7.17 ms at np2/4/8 vs campaign
docs/performance/scaling/levante_campaign_2026-07-24.md-139-19.90/17.09/6.92 — ~5% profiler overhead); (2) at np4 ONLY, giant
docs/performance/scaling/levante_campaign_2026-07-24.md-140-serialized "loop fusion" kernels appear — loop_add_fusion_1/2 at 3.6 ms
docs/performance/scaling/levante_campaign_2026-07-24.md-141-per launch (vs ~3 us for ordinary elementwise kernels) plus a THIRD
docs/performance/scaling/levante_campaign_2026-07-24.md-142-once-per-step group (the unsuffixed loop_add_fusion: 12 of its 44
--
docs/performance/scaling/levante_campaign_2026-07-24.md-146-wait): 3.6 + 3.6 + 3.3 ~= 10.5-10.9 ms/step = the np4 excess; (3) in the optimized
docs/performance/scaling/levante_campaign_2026-07-24.md-147-step HLO these are mega-fusions ON THE HALO PATH: `%loop_add_fusion =
docs/performance/scaling/levante_campaign_2026-07-24.md-148-f32[491520,26]` (edge-tendency add chain, 22 operands incl. an
docs/performance/scaling/levante_campaign_2026-07-24.md-149-input_scatter_fusion) and `%loop_add_fusion.4 = f32[163844,26]` (cell
docs/performance/scaling/levante_campaign_2026-07-24.md-150-array), with the shard_map halo-pack concatenates taking the same adds +
docs/performance/scaling/levante_campaign_2026-07-24.md:151:parameter lists as operands. PLAUSIBLE (inferred from kInput fusion
docs/performance/scaling/levante_campaign_2026-07-24.md-152-semantics + operand lists, not separately timed): the emitter RECOMPUTES
docs/performance/scaling/levante_campaign_2026-07-24.md-153-the expensive scatter+add chain inside each consumer fusion, which is why
docs/performance/scaling/levante_campaign_2026-07-24.md-154-the cost multiplies. WHY np4: fusion cost-model decisions depend on the
docs/performance/scaling/levante_campaign_2026-07-24.md-155-shard shape; at np2/np8 the mega-fusion is not built. This also explains
docs/performance/scaling/levante_campaign_2026-07-24.md-156-why the earlier env-knob sweep missed it — autotune/latency-hiding flags
--
docs/performance/scaling/levante_campaign_2026-07-24.md-211-multinode improvement measured at LL576 was partly a floor effect, and at
docs/performance/scaling/levante_campaign_2026-07-24.md-212-a production tile the identical code scales substantially better (0.37 ->
docs/performance/scaling/levante_campaign_2026-07-24.md-213-0.63). Per-device throughput also rises (259 -> 305 Mc/s/GPU at np4).
docs/performance/scaling/levante_campaign_2026-07-24.md-214-Config is byte-identical between the two rows; only the grid changes.
docs/performance/scaling/levante_campaign_2026-07-24.md-215-
docs/performance/scaling/levante_campaign_2026-07-24.md:216:REFUTED EN ROUTE: the np8 leg timed out twice (>90 min still tracing) while
docs/performance/scaling/levante_campaign_2026-07-24.md-217-np4 — a LARGER per-device tile — finished in ~25 min, which looked like a
docs/performance/scaling/levante_campaign_2026-07-24.md-218-compile-time cliff at that device count. It is not: the third attempt ran
docs/performance/scaling/levante_campaign_2026-07-24.md-219-the identical configuration in **99 seconds** with a 21.6 s compile (job
docs/performance/scaling/levante_campaign_2026-07-24.md-220-26457693). The earlier hangs were transient/environmental, not
docs/performance/scaling/levante_campaign_2026-07-24.md-221-reproducible, and no compile-time defect is claimed.
--
docs/performance/scaling/levante_campaign_2026-07-24.md-246-  differ in machine (A100-80 SXM vs A100-40), jax/tree version and date,
docs/performance/scaling/levante_campaign_2026-07-24.md-247-  so the ~13% gap is not attributable to any single factor.
docs/performance/scaling/levante_campaign_2026-07-24.md-248-- Cube "404 vs 141 Mc/s": the 404 is the single-GPU RTX-5090 Held-Suarez
docs/performance/scaling/levante_campaign_2026-07-24.md-249-  row in `SCALING_SUMMARY.md` SS1; ours is gray+SBM on A100 (job 26445836).
docs/performance/scaling/levante_campaign_2026-07-24.md-250-  That file's own tier table prices gray+SBM ~3x Held-Suarez, so ~135 is
docs/performance/scaling/levante_campaign_2026-07-24.md:251:  the expected equivalent vs 141 measured. PLAUSIBLE reconciliation from
docs/performance/scaling/levante_campaign_2026-07-24.md-252-  two published tables, NOT a matched A/B (GPU, physics and date differ).
docs/performance/scaling/levante_campaign_2026-07-24.md-253-- Ocean absolutes (A100 f64 165-201 Mc/s, f32 352, job 26445836; 5090 f64
docs/performance/scaling/levante_campaign_2026-07-24.md-254-  152 / f32 400 from `SCALING_SUMMARY.md` SS1) are of the same order -
docs/performance/scaling/levante_campaign_2026-07-24.md-255-  again a cross-machine sanity check, not a controlled comparison.
docs/performance/scaling/levante_campaign_2026-07-24.md-256-- Ginsburg "0.92 eff @2 GPU" reconciled: the old bench silently defaulted
--
docs/performance/scaling/levante_campaign_2026-07-24.md-350-Bound = per-device compute (32.58 / 14.63 ms) + modelled comm (2.21) +
docs/performance/scaling/levante_campaign_2026-07-24.md-351-modelled reduction (2.19). The unmodelled gap is **~5 ms/step and roughly
docs/performance/scaling/levante_campaign_2026-07-24.md-352-FLAT** with device count (7.9 ms at nd2, 6.7 at nd4), which is why the
docs/performance/scaling/levante_campaign_2026-07-24.md-353-ratio worsens as compute shrinks.
docs/performance/scaling/levante_campaign_2026-07-24.md-354-
docs/performance/scaling/levante_campaign_2026-07-24.md:355:WHAT THE GAP IS NOT — the omitted-traffic explanation is REFUTED
docs/performance/scaling/levante_campaign_2026-07-24.md-356-(`scripts/tmp/probe_ocean_halo_bytes.py`, HLO byte census on CPU virtual
docs/performance/scaling/levante_campaign_2026-07-24.md-357-devices). The bench's `comm_scope_note` correctly warns that its census is
docs/performance/scaling/levante_campaign_2026-07-24.md-358-"barotropic implicit-CN PCG scope only … baroclinic 3-D pads NOT counted",
docs/performance/scaling/levante_campaign_2026-07-24.md-359-and the true volume IS much larger: **16.22 MB/step across 110
docs/performance/scaling/levante_campaign_2026-07-24.md:360:collective-permutes vs the censused 2.25 MB — a 7.2x undercount**. But
docs/performance/scaling/levante_campaign_2026-07-24.md-361-completing the census moves the bound by only **0.22 ms**, because the
docs/performance/scaling/levante_campaign_2026-07-24.md-362-comm term is LATENCY-dominated: at 122 messages x 17.82 us the latency part
docs/performance/scaling/levante_campaign_2026-07-24.md-363-is 2.174 ms while even 16 MB at 64.22 GB/s is just 0.253 ms.
docs/performance/scaling/levante_campaign_2026-07-24.md-364-
docs/performance/scaling/levante_campaign_2026-07-24.md-365-So with the byte census completed the unexplained residual is still 5.5 ms
docs/performance/scaling/levante_campaign_2026-07-24.md-366-(nd2) and 4.3 ms (nd4).
docs/performance/scaling/levante_campaign_2026-07-24.md-367-
docs/performance/scaling/levante_campaign_2026-07-24.md:368:SECOND CANDIDATE ALSO REFUTED (`scripts/tmp/probe_sharded_overhead.py`,
docs/performance/scaling/levante_campaign_2026-07-24.md-369-job 26458553): the sharded formulation does NOT do measurably more work.
docs/performance/scaling/levante_campaign_2026-07-24.md-370-Timing the SHARDED step on a 1-device mesh (all the padding, band-edge and
docs/performance/scaling/levante_campaign_2026-07-24.md-371-v-row-reconstruction machinery present, ppermutes self-to-self so no real
docs/performance/scaling/levante_campaign_2026-07-24.md-372-traffic) against the UNSHARDED step at the identical tile:
docs/performance/scaling/levante_campaign_2026-07-24.md-373-
--
docs/performance/scaling/levante_campaign_2026-07-24.md-383-divided by the message count is **83 us/message at nd2 and 73 us at nd4**,
docs/performance/scaling/levante_campaign_2026-07-24.md-384-versus **17.8 us** for the same collective measured in isolation — an in-
docs/performance/scaling/levante_campaign_2026-07-24.md-385-context cost 4-5x the best case. That is consistent with EXPOSED,
docs/performance/scaling/levante_campaign_2026-07-24.md-386-un-overlapped communication rather than raw wire time.
docs/performance/scaling/levante_campaign_2026-07-24.md-387-
docs/performance/scaling/levante_campaign_2026-07-24.md:388:THIRD CANDIDATE REFUTED, AND IT IDENTIFIES THE MECHANISM (job 26458930).
docs/performance/scaling/levante_campaign_2026-07-24.md-389-If the residual were communication the scheduler is currently hiding work
docs/performance/scaling/levante_campaign_2026-07-24.md-390-behind, DISABLING XLA's latency-hiding scheduler would hurt. It does not:
docs/performance/scaling/levante_campaign_2026-07-24.md-391-
docs/performance/scaling/levante_campaign_2026-07-24.md-392-| arm | nd2 | nd4 | vs default |
docs/performance/scaling/levante_campaign_2026-07-24.md-393-|---|---|---|---|
--
docs/performance/scaling/levante_campaign_2026-07-24.md-454-genuinely PER-ITERATION, not a constant misattributed to iterations. But
docs/performance/scaling/levante_campaign_2026-07-24.md-455-the nd4 intercept EXCEEDS the measured 144-row compute term (14.63 ms) by
docs/performance/scaling/levante_campaign_2026-07-24.md-456-**1.48 ms**, so a fixed non-PCG overhead does exist and the 60 iterations
docs/performance/scaling/levante_campaign_2026-07-24.md-457-do NOT explain the entire residual.
docs/performance/scaling/levante_campaign_2026-07-24.md-458-
docs/performance/scaling/levante_campaign_2026-07-24.md:459:CODEX OBJECTION (a) TESTED AND REFUTED (job 26459817). Rather than divide
docs/performance/scaling/levante_campaign_2026-07-24.md-460-the nd1 slope by 4, measure the per-iteration slope directly at each tile
docs/performance/scaling/levante_campaign_2026-07-24.md-461-on ONE device:
docs/performance/scaling/levante_campaign_2026-07-24.md-462-
docs/performance/scaling/levante_campaign_2026-07-24.md-463-| tile | measured us/iter |
docs/performance/scaling/levante_campaign_2026-07-24.md-464-|---|---|
--
docs/performance/scaling/levante_campaign_2026-07-24.md-640-
docs/performance/scaling/levante_campaign_2026-07-24.md-641-Per-step time is flat 100 -> 600 steps (12.81 -> 12.74 ms). Both arms' heat
docs/performance/scaling/levante_campaign_2026-07-24.md-642-drift grows ~linearly and is similar between them, consistent with the
docs/performance/scaling/levante_campaign_2026-07-24.md-643-shared baroclinic/tracer path dominating it.
docs/performance/scaling/levante_campaign_2026-07-24.md-644-
docs/performance/scaling/levante_campaign_2026-07-24.md:645:MECHANISM CONFIRMED FROM A THIRD ANGLE. The wide-halo arm records its own
docs/performance/scaling/levante_campaign_2026-07-24.md-646-message census: **120 standard barotropic messages/step -> 4** (n_loop=30
docs/performance/scaling/levante_campaign_2026-07-24.md-647-substeps, stencil reach 3, one fixed wide exchange per chunk). A 30x cut in
docs/performance/scaling/levante_campaign_2026-07-24.md-648-barotropic exchanges is precisely why it wins where sync dominates, and it
docs/performance/scaling/levante_campaign_2026-07-24.md-649-is the SAME quantity the single_reduce analysis isolated as the half it
docs/performance/scaling/levante_campaign_2026-07-24.md-650-could not touch (44.1 us/iter of matvec halo). Three independent
--
docs/performance/scaling/levante_campaign_2026-07-24.md-982-
docs/performance/scaling/levante_campaign_2026-07-24.md-983-## Cube optimisation: bounded BEFORE implementing — and the bound killed the plan
docs/performance/scaling/levante_campaign_2026-07-24.md-984-
docs/performance/scaling/levante_campaign_2026-07-24.md-985-Directive was to push the cube toward its limit. Codex round-16 defined
docs/performance/scaling/levante_campaign_2026-07-24.md-986-the strategy first (per the standing pre-implementation rule) and its
docs/performance/scaling/levante_campaign_2026-07-24.md:987:cheapest-bound step then REFUTED the intervention I was about to build.
docs/performance/scaling/levante_campaign_2026-07-24.md-988-
docs/performance/scaling/levante_campaign_2026-07-24.md-989-**Codex corrections to my reading:**
docs/performance/scaling/levante_campaign_2026-07-24.md-990-* My "88 SendRecv / 4 rounds = 22 exchanges" was wrong. The kt=2 tile pad
docs/performance/scaling/levante_campaign_2026-07-24.md-991-  emits **16 phases per logical scalar pad** (4 edges + 4 guards + 4
docs/performance/scaling/levante_campaign_2026-07-24.md-992-  diagonals + 4 corner slivers) for serial-exact offset/corner handling
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1531-   CLOSED as not-worth-building, with receipts. (Also moot for the SPMD
docs/performance/scaling/levante_campaign_2026-07-24.md-1532-   lane: jax equal-shard sharding would need padding to the max band,
docs/performance/scaling/levante_campaign_2026-07-24.md-1533-   returning exactly the imbalance removed.)
docs/performance/scaling/levante_campaign_2026-07-24.md-1534-
docs/performance/scaling/levante_campaign_2026-07-24.md-1535-   *Expensive half — gather/scatter compaction: MEASURED, and the audit's
docs/performance/scaling/levante_campaign_2026-07-24.md:1536:   "~2x" is REFUTED* (`bench_gather_vs_slice_stencil.py`, job 26479884,
docs/performance/scaling/levante_campaign_2026-07-24.md-1537-   A100 f32, correctness self-checked). Per-cell gather penalty for a
docs/performance/scaling/levante_campaign_2026-07-24.md-1538-   5-point Laplacian vs the dense sliced version: **1.40-1.77x**, so
docs/performance/scaling/levante_campaign_2026-07-24.md-1539-   compaction wins only when wet_fraction < 0.56-0.72 (size-dependent).
docs/performance/scaling/levante_campaign_2026-07-24.md-1540-   At the REAL global-ocean wet fraction (~0.71), packed-gather is a net
docs/performance/scaling/levante_campaign_2026-07-24.md-1541-   LOSS on the full LL576 grid (ratio 1.16) and a wash at the nd4 tile
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1628-  METIS was NOT tested).
docs/performance/scaling/levante_campaign_2026-07-24.md-1629-* **`block:cyclic` stays mandatory on packed CPU lanes** (E/D = 1.61x at
docs/performance/scaling/levante_campaign_2026-07-24.md-1630-  a byte-identical partition). NOTE the second `--distribution` field is
docs/performance/scaling/levante_campaign_2026-07-24.md-1631-  the INTRA-NODE (socket) distribution — both arms place ranks on nodes
docs/performance/scaling/levante_campaign_2026-07-24.md-1632-  identically; the swing is socket-level. Mechanism (per-socket
docs/performance/scaling/levante_campaign_2026-07-24.md:1633:  memory-bandwidth balance) PLAUSIBLE, consistent with the np16 Milan
docs/performance/scaling/levante_campaign_2026-07-24.md-1634-  2.13x receipt; never instrumented with bandwidth counters.
docs/performance/scaling/levante_campaign_2026-07-24.md-1635-* Caveats: timing-only receipt — no parity/conservation gate ran in
docs/performance/scaling/levante_campaign_2026-07-24.md-1636-  these arms, and the CPU nodes emit `UCX WARN transports
docs/performance/scaling/levante_campaign_2026-07-24.md-1637-  'cuda_copy','cuda_ipc','gdr_copy' are not available` (the _env.sh GPU
docs/performance/scaling/levante_campaign_2026-07-24.md-1638-  UCX_TLS list on a CPU node; UCX falls back to rc/sm — cosmetic for
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1643-f32, sfc partition (padded-128 reorder), lloyd=0 LABELLED SYNTHETIC
docs/performance/scaling/levante_campaign_2026-07-24.md-1644-scaling mesh, executed padded n_cells = 2,621,568 (natural 2,621,442),
docs/performance/scaling/levante_campaign_2026-07-24.md-1645-L26; steps 12 / warmup 3; physics=none dynamics-only bench. Provenance:
docs/performance/scaling/levante_campaign_2026-07-24.md-1646-np64 and np128 rows record `git_sha: 7151d12a1`; the np32 row's field
docs/performance/scaling/levante_campaign_2026-07-24.md-1647-reads `unknown` — same allocation, same submitted script, so the same
docs/performance/scaling/levante_campaign_2026-07-24.md:1648:binary is PLAUSIBLE but that row stays non-reproduction-grade on its
docs/performance/scaling/levante_campaign_2026-07-24.md-1649-own (codex r20/r21):
docs/performance/scaling/levante_campaign_2026-07-24.md-1650-
docs/performance/scaling/levante_campaign_2026-07-24.md-1651-| GPUs | cells/GPU | ms/step | GC/s (cell-levels) |
docs/performance/scaling/levante_campaign_2026-07-24.md-1652-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-1653-| 32 | 81.9k | 12.47 | 5.47 |
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1678-   sets vs SAME-JOB solo controls bracketing phase B (solo before AND
docs/performance/scaling/levante_campaign_2026-07-24.md-1679-   after — BRACKETED, not fully counterbalanced; a penalty's attribution
docs/performance/scaling/levante_campaign_2026-07-24.md-1680-   to fabric vs placement/drift needs the per-step nodelist table +
docs/performance/scaling/levante_campaign_2026-07-24.md-1681-   follow-up). steps=5000 so the stepping window
docs/performance/scaling/levante_campaign_2026-07-24.md-1682-   (~60 s) dwarfs launch skew; per-arm `SLURM_STEP_NODELIST` +
docs/performance/scaling/levante_campaign_2026-07-24.md:1683:   wall-clock brackets logged as overlap evidence. CONFIRM bar:
docs/performance/scaling/levante_campaign_2026-07-24.md-1684-   max(replica) <= 1.10x mean(solo) => guaranteed aggregate >= 3.64x the
docs/performance/scaling/levante_campaign_2026-07-24.md-1685-   32-GPU solo rate (>= 19.9 GC/s if solo reproduces 5.47) = ~3.3x the
docs/performance/scaling/levante_campaign_2026-07-24.md:1686:   observed 128-GPU single-trajectory rate. REFUTE: replica slowdown
docs/performance/scaling/levante_campaign_2026-07-24.md-1687-   >10 % = a CO-EXECUTION penalty, quantified per replica — its
docs/performance/scaling/levante_campaign_2026-07-24.md-1688-   attribution (fabric contention vs placement/topology vs drift) is a
docs/performance/scaling/levante_campaign_2026-07-24.md-1689-   follow-up, not a conclusion of this job.
docs/performance/scaling/levante_campaign_2026-07-24.md-1690-2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
docs/performance/scaling/levante_campaign_2026-07-24.md-1691-   (81.9k/41.0k/20.5k cells/GPU) on the SAME lloyd=0 family, same sfc +
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1743-
docs/performance/scaling/levante_campaign_2026-07-24.md-1744-Distribution verified against the masquerade trap: result rows carry
docs/performance/scaling/levante_campaign_2026-07-24.md-1745-`n_ranks: 512` (the JSON's `metadata.process_count: 1` is the jax-LOCAL
docs/performance/scaling/levante_campaign_2026-07-24.md-1746-count on this mpi4jax lane, not the world size). 128->256 is
docs/performance/scaling/levante_campaign_2026-07-24.md-1747-SUPERLINEAR (2.23x for 2x) — classic per-rank working-set cache
docs/performance/scaling/levante_campaign_2026-07-24.md:1748:transition on Milan (mechanism PLAUSIBLE, uninstrumented). End-to-end
docs/performance/scaling/levante_campaign_2026-07-24.md-1749-64->512 eff 0.83 at 1k cols/rank: TIMING-ONLY evidence that the lat-lon
docs/performance/scaling/levante_campaign_2026-07-24.md-1750-CPU lane scales into the hundreds. QUALIFIER (codex r22): the run logs
docs/performance/scaling/levante_campaign_2026-07-24.md-1751-an out-of-tested-range mpi4jax==0.9.0 pairing ("may fail or produce
docs/performance/scaling/levante_campaign_2026-07-24.md-1752-incorrect results", parallel/reductions.py runtime check) and UCX
docs/performance/scaling/levante_campaign_2026-07-24.md-1753-VM_UNMAP warnings — no parity/conservation gate ran, so this ladder is
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1772-|---|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-1773-| 81.9k | np8 6.58 | np32 12.47 | 1.895 | **0.53** |
docs/performance/scaling/levante_campaign_2026-07-24.md-1774-| 41.0k | np16 6.43 | np64 9.60 | 1.493 | 0.67 |
docs/performance/scaling/levante_campaign_2026-07-24.md-1775-| 20.5k | np32 7.29 | np128 11.48 | 1.575 | 0.64 |
docs/performance/scaling/levante_campaign_2026-07-24.md-1776-
docs/performance/scaling/levante_campaign_2026-07-24.md:1777:* The falsifiability block's CONFIRM branch fires: ratios stay well
docs/performance/scaling/levante_campaign_2026-07-24.md-1778-  above 1 with the known Lloyd-family mismatch REMOVED. (This does not
docs/performance/scaling/levante_campaign_2026-07-24.md-1779-  prove the old confound "only" biased the size — these are
docs/performance/scaling/levante_campaign_2026-07-24.md-1780-  unreplicated single runs from separate allocations, one comparator
docs/performance/scaling/levante_campaign_2026-07-24.md-1781-  without row-level provenance; the confounded draft read
docs/performance/scaling/levante_campaign_2026-07-24.md-1782-  1.80/1.35/1.41 vs 1.90/1.49/1.57 here, and the production-mesh s8
docs/performance/scaling/levante_campaign_2026-07-24.md-1783-  np8 was 6.92 vs lloyd-0 6.58, -4.9 %, so the mesh family does shift
docs/performance/scaling/levante_campaign_2026-07-24.md-1784-  absolutes.)
docs/performance/scaling/levante_campaign_2026-07-24.md-1785-* Restated: at NEAR-matched per-GPU tile, ~quadrupling devices+problem
docs/performance/scaling/levante_campaign_2026-07-24.md-1786-  costs 1.5-1.9x on this lane — the GPU-side analogue of the ocean CPU
docs/performance/scaling/levante_campaign_2026-07-24.md-1787-  scale-out term. Weak efficiency 0.53-0.67 at 4x. Mechanism still
docs/performance/scaling/levante_campaign_2026-07-24.md:1788:  UNATTRIBUTED (PLAUSIBLE candidates unchanged: inter-node neighbour
docs/performance/scaling/levante_campaign_2026-07-24.md-1789-  fraction growth, collective latency vs count, sfc partition-quality
docs/performance/scaling/levante_campaign_2026-07-24.md-1790-  decay with parts; the metis receipt argues against pure
docs/performance/scaling/levante_campaign_2026-07-24.md-1791-  partition-cut explanations, on the CPU lane at least).
docs/performance/scaling/levante_campaign_2026-07-24.md-1792-* The non-monotone tile dependence of the ratio (largest at the
docs/performance/scaling/levante_campaign_2026-07-24.md-1793-  LARGEST tile, 1.90 at 81.9k) is unexplained; recorded, not theorised.
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1797-Closed the bench's own honest-null bound gap (audit item 4) for the
docs/performance/scaling/levante_campaign_2026-07-24.md-1798-lat-lon lane, using only repo instruments:
docs/performance/scaling/levante_campaign_2026-07-24.md-1799-
docs/performance/scaling/levante_campaign_2026-07-24.md-1800-* **Halo census** (new probe `scripts/tmp/_probe_latlon_halo_census.py`,
docs/performance/scaling/levante_campaign_2026-07-24.md-1801-  virtual-CPU forced-host-platform lowering of the REAL
docs/performance/scaling/levante_campaign_2026-07-24.md:1802:  `make_sharded_atm_latlon_step`): **41 collective-permutes + 1
docs/performance/scaling/levante_campaign_2026-07-24.md-1803-  all-reduce per step**, nd-INDEPENDENT (identical at nd=8 and nd=16 —
docs/performance/scaling/levante_campaign_2026-07-24.md-1804-  the 1-D band structure check). Exact CP payload from compiled-HLO
docs/performance/scaling/levante_campaign_2026-07-24.md-1805-  result shapes: 4,635,408 B/dev/step at n_lon=1024 L26 f32 = 1.06x the
docs/performance/scaling/levante_campaign_2026-07-24.md-1806-  single-row slab model; linear in n_lon (checked 1024 vs 2048, 0.07 %
docs/performance/scaling/levante_campaign_2026-07-24.md-1807-  residual) -> **18.5 MB/dev/step at n_lon=4096 f32**. CAVEAT: CPU
docs/performance/scaling/levante_campaign_2026-07-24.md-1808-  lowering; GPU-side collective combining could change the executed
docs/performance/scaling/levante_campaign_2026-07-24.md-1809-  count (metadata.py:214) — the bound is a MODEL.
docs/performance/scaling/levante_campaign_2026-07-24.md:1810:* **Same-tile nd=1 compute baselines** (job 26630370, roofline recipe):
docs/performance/scaling/levante_campaign_2026-07-24.md-1811-  16x4096 f32 1.659 ms, 32x4096 f32 2.837, 16x4096 f64 2.973.
docs/performance/scaling/levante_campaign_2026-07-24.md-1812-  Approximation, recorded: nd=1 includes pole tiles -> compute term
docs/performance/scaling/levante_campaign_2026-07-24.md-1813-  biased HIGH -> bound conservative.
docs/performance/scaling/levante_campaign_2026-07-24.md:1814:* **Calibrated bound** (`metadata.calibrated_bound`, measured fabric
docs/performance/scaling/levante_campaign_2026-07-24.md-1815-  constants: IB 26.3 us / 23.5 GB/s, NVLink 17.8 / 64.2):
docs/performance/scaling/levante_campaign_2026-07-24.md-1816-
docs/performance/scaling/levante_campaign_2026-07-24.md-1817-| row | measured | t_bound (IB) | measured/bound |
docs/performance/scaling/levante_campaign_2026-07-24.md-1818-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-1819-| LL2048@64 f32 | 6.732 | 2.863 | **2.35** |
docs/performance/scaling/levante_campaign_2026-07-24.md-1820-| LL2048@128 f32 | 5.577 | 1.848 | **3.02** |
docs/performance/scaling/levante_campaign_2026-07-24.md-1821-| LL2048@128 f64 | 9.602 | 2.999 | **3.20** |
docs/performance/scaling/levante_campaign_2026-07-24.md-1822-
docs/performance/scaling/levante_campaign_2026-07-24.md:1823:* **The lat-lon GPU panel sits ~2.4-3.2x ABOVE its modeled limit** — the
docs/performance/scaling/levante_campaign_2026-07-24.md-1824-  eff-0.60 strong leg is NOT close to the fabric+compute floor. Leading
docs/performance/scaling/levante_campaign_2026-07-24.md:1825:  PLAUSIBLE mechanism (uninstrumented): effective per-CP cost
docs/performance/scaling/levante_campaign_2026-07-24.md-1826-  (launch + schedule + stream sync) well above the raw 26 us fabric
docs/performance/scaling/levante_campaign_2026-07-24.md-1827-  latency across 41 dependency-chained exchanges — the arXiv:2607.16100
docs/performance/scaling/levante_campaign_2026-07-24.md-1828-  small-collective regime. The model itself notes the serialized-latency
docs/performance/scaling/levante_campaign_2026-07-24.md-1829-  vs overlap biases pull opposite ways; treat measured/bound as a
docs/performance/scaling/levante_campaign_2026-07-24.md-1830-  consistency diagnostic, not proven headroom.
docs/performance/scaling/levante_campaign_2026-07-24.md:1831:* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
docs/performance/scaling/levante_campaign_2026-07-24.md-1832-  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
docs/performance/scaling/levante_campaign_2026-07-24.md-1833-  same-job control, falsifiability block in the script. The ocean-lane
docs/performance/scaling/levante_campaign_2026-07-24.md-1834-  null for these flags was reduction-dominated — first atm test.
--
./CMIP.md-376-  atm surface layer gets) → interface flux is energy-CONSISTENT.
./CMIP.md-377-
./CMIP.md-378-Validation: 15d A/B job 8558650 (vs prior 10d 8554291: R_TOA −27, CWV 17.6,
./CMIP.md-379-drift −133) — expect higher CWV, less-negative R_TOA, arrested drift.
./CMIP.md-380-
./CMIP.md:381:**Gustiness alone CONFIRMED insufficient** (15d job 8558910): R_TOA −54,
./CMIP.md-382-SST drift −92 K/yr, column-T 256.8→250.9 K monotonic, circulation dying. The
./CMIP.md-383-standard complement — WOA surface T/S restoring — was implemented (commit
./CMIP.md-384-b071ba359): canonical Haney kernel `ocean/forcing/surface_relaxation.py`
./CMIP.md-385-(run_omip's `_apply_restoring` refactored onto it; no duplicate numerics),
./CMIP.md-386-`CoupledConfig.ocean_restore_{sst,sss}_tau_days`, `_apply_ocean_restoring` in
--
./tests/bench/test_scaling_metadata.py-365-def test_count_collective_permutes_matches_hyphen_and_underscore():
./tests/bench/test_scaling_metadata.py-366-    """Canonical CP census (#1113): counts StableHLO underscore + optimized
./tests/bench/test_scaling_metadata.py-367-    hyphen spellings, and drops the async ``-done`` companion so one logical
./tests/bench/test_scaling_metadata.py-368-    exchange counts once."""
./tests/bench/test_scaling_metadata.py-369-    hlo = "\n".join([
./tests/bench/test_scaling_metadata.py:370:        "  %a = collective-permute(%x)",          # optimized HLO
./tests/bench/test_scaling_metadata.py-371-        "  %b = collective_permute(%y)",          # StableHLO
./tests/bench/test_scaling_metadata.py:372:        "  %c = collective-permute-done(%a)",     # async companion -> excluded
./tests/bench/test_scaling_metadata.py-373-        "  %d = collective_permute_done(%b)",     # async companion -> excluded
./tests/bench/test_scaling_metadata.py-374-        "  %e = all-gather(%z)",                  # different collective
./tests/bench/test_scaling_metadata.py-375-    ])
./tests/bench/test_scaling_metadata.py-376-    assert md.count_collective_permutes(hlo) == 2
./tests/bench/test_scaling_metadata.py-377-    assert md.count_collective_permutes("no collectives here") == 0
--
./tests/bench/test_scaling_metadata.py-392-    import jax
./tests/bench/test_scaling_metadata.py-393-    return jax.default_device(jax.devices("cpu")[0])
./tests/bench/test_scaling_metadata.py-394-
./tests/bench/test_scaling_metadata.py-395-
./tests/bench/test_scaling_metadata.py-396-def test_hlo_collective_permutes_lowers_counts_and_is_error_safe():
./tests/bench/test_scaling_metadata.py:397:    """The best-effort probe lowers a fn and counts its collective-permutes: a
./tests/bench/test_scaling_metadata.py-398-    fn with none -> 0; an unlowerable fn -> None (never raises). Real ppermute
./tests/bench/test_scaling_metadata.py-399-    counting is exercised by the MPAS/cube bench gates and
./tests/bench/test_scaling_metadata.py-400-    count_collective_permutes' synthetic HLO test above."""
./tests/bench/test_scaling_metadata.py-401-    import jax.numpy as jnp
./tests/bench/test_scaling_metadata.py-402-    with _cpu_compile():
--
./tests/bench/test_scaling_metadata.py-413-    underscore + optimized hyphen, and a config-header flag echo that merely
./tests/bench/test_scaling_metadata.py-414-    CONTAINS an op name never inflates the count."""
./tests/bench/test_scaling_metadata.py-415-    hlo = "\n".join([
./tests/bench/test_scaling_metadata.py-416-        # config-header echo of XLA_FLAGS -> must NOT match (no op-call paren)
./tests/bench/test_scaling_metadata.py-417-        "  // xla_gpu_collective_permute_combine_threshold_bytes=33554432",
./tests/bench/test_scaling_metadata.py:418:        "  %a = collective-permute(%x)",             # permute (optimized)
./tests/bench/test_scaling_metadata.py-419-        "  %b = collective_permute(%y)",             # permute (StableHLO)
./tests/bench/test_scaling_metadata.py:420:        "  %c = collective-permute-done(%a)",        # async companion -> drop
./tests/bench/test_scaling_metadata.py-421-        "  %r1 = all-reduce(%p)",                    # reduction (hyphen)
./tests/bench/test_scaling_metadata.py-422-        "  %r2 = all_reduce_start(%q)",              # async reduction -> count once
./tests/bench/test_scaling_metadata.py-423-        "  %r3 = all-reduce-done(%r2)",              # async companion -> drop
./tests/bench/test_scaling_metadata.py-424-        "  %g = all-gather(%z)",                     # all-gather
./tests/bench/test_scaling_metadata.py-425-        "  %a2a = all-to-all(%w)",                   # all-to-all
--
./tests/bench/test_scaling_metadata.py-447-    a line that merely CONTAINS the substring "done" elsewhere — an XLA
./tests/bench/test_scaling_metadata.py-448-    metadata op_name, a ``%done_*`` SSA name — must STILL be counted.  A blunt
./tests/bench/test_scaling_metadata.py-449-    ``"done" not in line`` filter would false-drop these to zero."""
./tests/bench/test_scaling_metadata.py-450-    hlo = "\n".join([
./tests/bench/test_scaling_metadata.py-451-        '  %r = all-reduce(%p), metadata={op_name="jit(step)/done_stage/psum"}',
./tests/bench/test_scaling_metadata.py:452:        '  %done_mass = f32[] collective-permute(%q)',
./tests/bench/test_scaling_metadata.py-453-        '  %g = all-gather(%z), metadata={op_name="reduce_done/x"}',
./tests/bench/test_scaling_metadata.py-454-    ])
./tests/bench/test_scaling_metadata.py-455-    c = md.count_collectives(hlo)
./tests/bench/test_scaling_metadata.py-456-    assert c["all_reduce"] == 1        # NOT dropped despite "done" in metadata
./tests/bench/test_scaling_metadata.py-457-    assert c["collective_permute"] == 1  # NOT dropped despite %done_ SSA name
--
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
--
./tests/bench/test_m1_accounting.py-1-"""M1 increment-2 accounting helpers (audit items 4/8/9): pure arithmetic.
./tests/bench/test_m1_accounting.py-2-
./tests/bench/test_m1_accounting.py-3-Direct unit tests for ``scripts/bench/metadata.py``'s
./tests/bench/test_m1_accounting.py:4:``calibrated_bound`` / ``comm_accounting`` / ``wet_cell_metrics`` — the
./tests/bench/test_m1_accounting.py-5-honest-null contract is the point: a missing ingredient must surface as
./tests/bench/test_m1_accounting.py-6-``None`` + a named reason, NEVER a fabricated number.  CPU/pure-python
./tests/bench/test_m1_accounting.py-7-(the helpers import no JAX).  The end-to-end JSONL wiring of these fields
./tests/bench/test_m1_accounting.py-8-is asserted by ``test_bench_ocean_latlon_spmd_gates.py``'s smoke run.
./tests/bench/test_m1_accounting.py-9-"""
--
./tests/bench/test_m1_accounting.py-17-
./tests/bench/test_m1_accounting.py-18-sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench"))
./tests/bench/test_m1_accounting.py-19-from metadata import (  # noqa: E402
./tests/bench/test_m1_accounting.py-20-    DEFAULT_COMM_BANDWIDTH_GBS,
./tests/bench/test_m1_accounting.py-21-    DEFAULT_COMM_LATENCY_US,
./tests/bench/test_m1_accounting.py:22:    calibrated_bound,
./tests/bench/test_m1_accounting.py-23-    comm_accounting,
./tests/bench/test_m1_accounting.py-24-    wet_cell_metrics,
./tests/bench/test_m1_accounting.py-25-)
./tests/bench/test_m1_accounting.py-26-
./tests/bench/test_m1_accounting.py-27-
./tests/bench/test_m1_accounting.py-28-# --------------------------------------------------------------------------
./tests/bench/test_m1_accounting.py:29:# calibrated_bound (audit item 8)
./tests/bench/test_m1_accounting.py-30-# --------------------------------------------------------------------------
./tests/bench/test_m1_accounting.py-31-
./tests/bench/test_m1_accounting.py-32-def test_bound_complete_arithmetic():
./tests/bench/test_m1_accounting.py-33-    """Hand-checked: T_bound = max(compute, comm) + reduction + launch.
./tests/bench/test_m1_accounting.py-34-
./tests/bench/test_m1_accounting.py-35-    Rank imbalance is a DIAGNOSTIC ingredient only — deliberately NOT
./tests/bench/test_m1_accounting.py-36-    added (the measured max/median ratio already contains comm/reduction
./tests/bench/test_m1_accounting.py-37-    jitter; adding it double-counts — codex batch4)."""
./tests/bench/test_m1_accounting.py:38:    out = calibrated_bound(
./tests/bench/test_m1_accounting.py-39-        measured_fused_step_ms=16.0,
./tests/bench/test_m1_accounting.py-40-        single_device_fused_step_ms=10.0,
./tests/bench/test_m1_accounting.py-41-        halo_messages_per_step=100,
./tests/bench/test_m1_accounting.py-42-        halo_bytes_per_step=1_000_000,
./tests/bench/test_m1_accounting.py-43-        n_reductions_per_step=50,
--
./tests/bench/test_m1_accounting.py-59-    assert out["bound_incomplete_reason"] is None
./tests/bench/test_m1_accounting.py-60-
./tests/bench/test_m1_accounting.py-61-
./tests/bench/test_m1_accounting.py-62-def test_bound_comm_dominates_via_max():
./tests/bench/test_m1_accounting.py-63-    """When comm > compute the overlappable max picks comm, not the sum."""
./tests/bench/test_m1_accounting.py:64:    out = calibrated_bound(
./tests/bench/test_m1_accounting.py-65-        measured_fused_step_ms=None,
./tests/bench/test_m1_accounting.py-66-        single_device_fused_step_ms=1.0,
./tests/bench/test_m1_accounting.py-67-        halo_messages_per_step=1000,
./tests/bench/test_m1_accounting.py-68-        halo_bytes_per_step=0,
./tests/bench/test_m1_accounting.py-69-        n_reductions_per_step=0,
--
./tests/bench/test_m1_accounting.py-74-    assert out["t_bound_ms"] == pytest.approx(10.0)
./tests/bench/test_m1_accounting.py-75-    assert out["measured_over_bound"] is None   # no measurement passed
./tests/bench/test_m1_accounting.py-76-
./tests/bench/test_m1_accounting.py-77-
./tests/bench/test_m1_accounting.py-78-def test_bound_missing_single_device_is_null_and_named():
./tests/bench/test_m1_accounting.py:79:    out = calibrated_bound(
./tests/bench/test_m1_accounting.py-80-        measured_fused_step_ms=8.0,
./tests/bench/test_m1_accounting.py-81-        single_device_fused_step_ms=None,       # the nd=1 row not provided
./tests/bench/test_m1_accounting.py-82-        halo_messages_per_step=10,
./tests/bench/test_m1_accounting.py-83-        halo_bytes_per_step=100,
./tests/bench/test_m1_accounting.py-84-        n_reductions_per_step=5,
--
./tests/bench/test_m1_accounting.py-97-def test_bound_placeholder_fabric_with_nonzero_comm_is_null():
./tests/bench/test_m1_accounting.py-98-    """All measurable ingredients present but the fabric is the placeholder
./tests/bench/test_m1_accounting.py-99-    AND there is real comm work: the bound must be an honest null (a
./tests/bench/test_m1_accounting.py-100-    placeholder latency/bandwidth would fabricate the number — codex
./tests/bench/test_m1_accounting.py-101-    batch4), with the missing calibration NAMED."""
./tests/bench/test_m1_accounting.py:102:    out = calibrated_bound(
./tests/bench/test_m1_accounting.py-103-        measured_fused_step_ms=8.0,
./tests/bench/test_m1_accounting.py-104-        single_device_fused_step_ms=4.0,
./tests/bench/test_m1_accounting.py-105-        halo_messages_per_step=10,
./tests/bench/test_m1_accounting.py-106-        halo_bytes_per_step=100,
./tests/bench/test_m1_accounting.py-107-        n_reductions_per_step=5,
--
./tests/bench/test_m1_accounting.py-117-
./tests/bench/test_m1_accounting.py-118-
./tests/bench/test_m1_accounting.py-119-def test_bound_placeholder_fabric_with_zero_comm_still_computes():
./tests/bench/test_m1_accounting.py-120-    """Zero comm/reduction work is fabric-INDEPENDENT (placeholders multiply
./tests/bench/test_m1_accounting.py-121-    zero): the nd=1 compute-only bound stays available uncalibrated."""
./tests/bench/test_m1_accounting.py:122:    out = calibrated_bound(
./tests/bench/test_m1_accounting.py-123-        measured_fused_step_ms=5.0,
./tests/bench/test_m1_accounting.py-124-        single_device_fused_step_ms=5.0,
./tests/bench/test_m1_accounting.py-125-        halo_messages_per_step=0,
./tests/bench/test_m1_accounting.py-126-        halo_bytes_per_step=0,
./tests/bench/test_m1_accounting.py-127-        n_reductions_per_step=0,
--
./tests/bench/test_m1_accounting.py-131-    assert out["bound_calibrated"] is False
./tests/bench/test_m1_accounting.py-132-    assert out["bound_incomplete_reason"] is None
./tests/bench/test_m1_accounting.py-133-
./tests/bench/test_m1_accounting.py-134-
./tests/bench/test_m1_accounting.py-135-def test_bound_missing_comm_census_is_null_and_named():
./tests/bench/test_m1_accounting.py:136:    out = calibrated_bound(
./tests/bench/test_m1_accounting.py-137-        measured_fused_step_ms=8.0,
./tests/bench/test_m1_accounting.py-138-        single_device_fused_step_ms=4.0,
./tests/bench/test_m1_accounting.py-139-        halo_messages_per_step=None,            # no census for this config
./tests/bench/test_m1_accounting.py-140-        halo_bytes_per_step=None,
./tests/bench/test_m1_accounting.py-141-        n_reductions_per_step=None,
--
./tests/bench/test_m1_accounting.py-147-        assert name in out["bound_incomplete_reason"]
./tests/bench/test_m1_accounting.py-148-
./tests/bench/test_m1_accounting.py-149-
./tests/bench/test_m1_accounting.py-150-def test_bound_single_device_trivial():
./tests/bench/test_m1_accounting.py-151-    """nd=1: zero comm is a fact -> bound == compute, ratio == 1."""
./tests/bench/test_m1_accounting.py:152:    out = calibrated_bound(
./tests/bench/test_m1_accounting.py-153-        measured_fused_step_ms=5.0,
./tests/bench/test_m1_accounting.py-154-        single_device_fused_step_ms=5.0,
./tests/bench/test_m1_accounting.py-155-        halo_messages_per_step=0,
./tests/bench/test_m1_accounting.py-156-        halo_bytes_per_step=0,
./tests/bench/test_m1_accounting.py-157-        n_reductions_per_step=0,
--
./tests/bench/test_m1_accounting.py-162-    assert out["bound_incomplete_reason"] is None
./tests/bench/test_m1_accounting.py-163-
./tests/bench/test_m1_accounting.py-164-
./tests/bench/test_m1_accounting.py-165-def test_bound_partial_calibration_not_calibrated():
./tests/bench/test_m1_accounting.py-166-    """Passing only ONE of latency/bandwidth still uses a placeholder."""
./tests/bench/test_m1_accounting.py:167:    out = calibrated_bound(
./tests/bench/test_m1_accounting.py-168-        single_device_fused_step_ms=1.0,
./tests/bench/test_m1_accounting.py-169-        halo_messages_per_step=0, halo_bytes_per_step=0,
./tests/bench/test_m1_accounting.py-170-        n_reductions_per_step=0, rank_imbalance=1.0,
./tests/bench/test_m1_accounting.py-171-        latency_us=3.0,           # bandwidth left to the placeholder
./tests/bench/test_m1_accounting.py-172-    )
./tests/bench/test_m1_accounting.py-173-    assert out["bound_calibrated"] is False
./tests/bench/test_m1_accounting.py-174-
./tests/bench/test_m1_accounting.py-175-
./tests/bench/test_m1_accounting.py-176-def test_bound_rejects_bad_fabric_numbers():
./tests/bench/test_m1_accounting.py-177-    with pytest.raises(ValueError):
./tests/bench/test_m1_accounting.py:178:        calibrated_bound(bandwidth_GBs=0.0, latency_us=1.0)
./tests/bench/test_m1_accounting.py-179-    with pytest.raises(ValueError):
./tests/bench/test_m1_accounting.py:180:        calibrated_bound(bandwidth_GBs=1.0, latency_us=-1.0)
./tests/bench/test_m1_accounting.py-181-
./tests/bench/test_m1_accounting.py-182-
./tests/bench/test_m1_accounting.py-183-# --------------------------------------------------------------------------
./tests/bench/test_m1_accounting.py-184-# comm_accounting (audit item 4)
./tests/bench/test_m1_accounting.py-185-# --------------------------------------------------------------------------
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-76-# under rec["metadata"] so a virtual-CPU-device proxy, a gloo/TCP fabric run,
scripts/bench/bench_atm_latlon_spmd_scaling.py-77-# or an f32 ablation is falsifiable from the JSONL row alone.  metadata.py
scripts/bench/bench_atm_latlon_spmd_scaling.py-78-# imports JAX lazily, so this is safe before jax.distributed.initialize.
scripts/bench/bench_atm_latlon_spmd_scaling.py-79-from metadata import (  # noqa: E402
scripts/bench/bench_atm_latlon_spmd_scaling.py-80-    annotate_incomplete,
scripts/bench/bench_atm_latlon_spmd_scaling.py:81:    calibrated_bound,
scripts/bench/bench_atm_latlon_spmd_scaling.py-82-    comm_accounting,
scripts/bench/bench_atm_latlon_spmd_scaling.py-83-    scaling_metadata,
scripts/bench/bench_atm_latlon_spmd_scaling.py-84-    tidy_throughput_fields,
scripts/bench/bench_atm_latlon_spmd_scaling.py-85-)
scripts/bench/bench_atm_latlon_spmd_scaling.py-86-
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-341-        bytes_per_message=_bytes_msg,
scripts/bench/bench_atm_latlon_spmd_scaling.py-342-        full_state_gathers_per_step=0,   # fused scan/segment: no per-step gather
scripts/bench/bench_atm_latlon_spmd_scaling.py-343-        scope_note=_comm_note,
scripts/bench/bench_atm_latlon_spmd_scaling.py-344-        bytes_are_lower_bound=_bytes_lower,
scripts/bench/bench_atm_latlon_spmd_scaling.py-345-    )
scripts/bench/bench_atm_latlon_spmd_scaling.py:346:    bound_rec = calibrated_bound(
scripts/bench/bench_atm_latlon_spmd_scaling.py-347-        measured_fused_step_ms=_measured_med,
scripts/bench/bench_atm_latlon_spmd_scaling.py-348-        # Invalid rows feed the bound NOTHING: even the CLI-provided nd=1
scripts/bench/bench_atm_latlon_spmd_scaling.py-349-        # baseline is withheld so bound_ingredients.compute_ms cannot dress
scripts/bench/bench_atm_latlon_spmd_scaling.py-350-        # a diverging row up as a modelled one (codex).
scripts/bench/bench_atm_latlon_spmd_scaling.py-351-        single_device_fused_step_ms=(
--
./docs/ocean/fidelity/oracle_recipe_strategy.md-538-|---|---|---|---|---|
./docs/ocean/fidelity/oracle_recipe_strategy.md-539-| **Flux-form momentum advection** — Veros uses flux-form ∇·(uu); legoESM had only `vector_invariant`/`weno`. | **yes** | Missing **method** | **DONE (built + adopted).** Added `momentum_advection="flux_form"` (+ `momentum_flux_scheme` upwind/centered) to the canonical dycore (reuses `divergence_cgrid` + face-interp); verified on truth tiers F1–F8 (bit-identity, conservation machine-eps, zero-velocity, uniform-flow, differentiability, gyre stability). Adopted `flux_form`/`centered` in `build_acc_model_config`. **Oracle (developed-flow) result: du_adv corr 0.584→0.654, dv_adv −0.336→0.715 vs Veros** — flux-form matches Veros's flux-form far better (apples-to-apples validated). See `flux_form_*` docs. | **DONE** |
./docs/ocean/fidelity/oracle_recipe_strategy.md-540-| **EKE closure** — Veros eddy-kinetic-energy parameterization (`enable_eke=True`, `eke_c_k=0.4`, superbee advection + isopycnal diffusion); no legoESM equivalent. | **yes** | Missing **method** | **DONE (built + oracle-verified + ADOPTED).** (Adoption was unblocked by the `eke_len` Rhines variant — see the next row.) Added the Eden-Greatbatch prognostic-EKE closure as a canonical block: `ocean/physics/lateral_mixing/eke.py` (2-D depth-integrated `E`; `kappa_GM=c_k·L·√E`; semi-implicit positivity; conservative transport) + `GMRediConfig.eke` + step coupling + restart. Truth tiers E1–E8 green (budget closure machine-eps, positivity-by-construction, differentiability, idealized-channel bounded spin-up, regression lock). **Oracle (E9, developed-flow Veros ACC, enable_eke): legoESM's `eke_kappa_gm` reproduces Veros's `K_gm` to MACHINE PRECISION (corr=1.0, max_rel_err=0 on 19560 wet cells) when fed Veros's own `eke`+`eke_len`** — the closure form + ACC params (c_k=0.4, c_eps=0.5, k_max=1e4, lmin=100, superbee, isopycnal diffusion, all verified against `veros/setups/acc/acc.py`) match exactly. See the EKE build docs + `eke_build_progress.md`. | **DONE** |
./docs/ocean/fidelity/oracle_recipe_strategy.md-541-| **EKE mixing length `eke_len`** — Veros: `max(lmin, min(eke_cross·L_rossby, eke_crhin·L_rhines))` with the eddy-energy-dependent **Rhines scale** `L_rhines=√(√E/β)` + `eke_cross=2` (developed ACC: ~8 km mean, 47 km max). legoESM's EKE `L` reuses the Visbeck first-baroclinic Rossby radius (~200 km, saturated) — ~25× larger. | **yes** (part of EKE) | Missing **variant** | **DONE (built + adopted).** Added the Rhines-limited mixing length as the selectable `EKEConfig.mixing_length_scheme="rhines"` (default `"rossby"` keeps the Visbeck length bit-identical): `eke_rhines_length=√(√E/β)`, `eke_deformation_radius=min(c₁/|f|, √(c₁/2β))` with `c₁=∫N dz/π` (reuses the shared `_eady_growth_and_length` N — no duplicate numerics), `eke_len_composite=max(lmin, min(eke_cross·L_def, eke_crhin·L_rhines))`; `β=2Ω cosφ/R` analytic from config constants. Truth tiers L1–L3 green (formula/dispatch unit tests, default bit-identical incl. E8 + decomposition goldens, differentiability finite+nonzero, dry-column NaN-safe). **Oracle (L4, developed-flow Veros ACC): fed Veros's own N²/f/β/eke, legoESM reproduces Veros `L_rossby` (5.8e-16), `L_rhines` (0.0), `eke_len` (0.0) AND `K_gm` (0.0) to MACHINE PRECISION** — Veros `eke_len` mean 8.3 km / max 47.3 km, the Rhines scale IS the limiter vs the 56.9 km mean deformation radius. ADOPTED in `ACC_GM_REDI_CONFIG` (`eke=EKEConfig(mixing_length_scheme="rhines", eke_cross=2.0, eke_crhin=1.0)`; initial `eke` field seeded so the scan-based free-run carries a constant pytree). See `eke_len_build_progress.md` + `eke_len_build_spec.md`. | **DONE** |
./docs/ocean/fidelity/oracle_recipe_strategy.md-542-| **EKE isopycnal diffusion (`K_iso = K_gm`)** — Veros ACC sets `enable_eke_isopycnal_diffusion=True` so the Redi *tracer* diffusivity follows the prognostic GM coefficient (`K_iso = K_gm`, ~0.3 m²/s cold-start; `veros/core/eke.py:74-75`). legoESM holds `kappa_Redi=1000` constant — only the GM *skew* term is EKE-driven, so during a cold start legoESM `K_iso=1000 ≫` Veros `~0.3` (a ~3000× mismatch). | **yes** (part of EKE) | Missing **method** | **Build:** make `kappa_Redi` array-capable through the GM/Redi triad+centered flux builders (must stay bit-identical on the constant-kappa default — golden-locked core numerics), add an `EKEConfig` iso-diffusion flag (Veros `enable_eke_isopycnal_diffusion`), and have the step pass `kappa_redi_override = kappa_gm_override` when active. Then regenerate the rhines step golden + re-measure the ACC free-run (the 30-day transport/KE gap interpretation depends on it). Surfaced by the eke_len **L5+L6 review (Issue 1)**. | **DONE (built + adopted).** Made `kappa_Redi` array-capable through the GM/Redi triad + centered flux builders (scalar default bit-identical — decomposition + 3 EKE goldens hold; gate R1, 125 tests) via the same face-interpolation as `kappa_GM`; added `EKEConfig.isopycnal_diffusion` (Veros `enable_eke_isopycnal_diffusion`, default False); the step passes `kappa_redi_override = kappa_gm_override` when set, so K_iso = K_gm and the `(kappa_Redi-kappa_GM)` horizontal off-diagonal cancels exactly as in Veros. ADOPTED in `ACC_GM_REDI_CONFIG` (`isopycnal_diffusion=True`). **Oracle (R3): Veros's own K_iso == K_gm to machine precision (max_rel_err 0.0, mean 33 m²/s); legoESM reproduces it transitively** (K_iso = kappa_redi_override = kappa_gm_override, the prognostic kappa that L4 matched to Veros K_gm machine-exact). **Measure-first (R3, EKE+Redi-on 30-day free-run): the ACC-transport/KE gap is UNCHANGED (+70.3%/+233.2%)** — with the WHOLE EKE/GM-Redi closure now apples-to-apples, the 30-day gap is conclusively the **integrator** (forward-Euler vs Veros AB2 — see the next row) + spin-up, NOT the eddy closure. Locked by the `rhines_kiso` step golden. | **DONE** |
./docs/ocean/fidelity/oracle_recipe_strategy.md:543:| **Outer time integrator** (audit gap #7) — legoESM forward-Euler / split-explicit vs Veros. **CORRECTION (2026-05-29, from Veros source): this Veros version uses Adams-Bashforth-2, NOT leapfrog+Robert-Asselin** — tracers `temp[taup1]=temp[tau]+dt_tracer·((1.5+ε)·dtemp[tau]−(0.5+ε)·dtemp[taum1])` (`thermodynamics.py`), AB2 momentum (`solve_stream.py`), separate `dt_tracer=43200`/`dt_mom=4800`, and the AB2 ε-offset (`AB_eps=0.1`) — NOT an Asselin filter — suppresses the computational mode. The audit doc's "leapfrog+AB2+Robert-Asselin" was wrong. With the WHOLE EKE/GM-Redi closure now apples-to-apples (machine-exact), the measure-first 30-day ACC gap (+70%/+233%) is conclusively dominated by THIS. | **yes** | Missing **method** | **Build the AB2 OUTER scheme** reusing the existing `timestepping/leapfrog_ab2.py:ab2_step` + the inner `tracer_time_integrator="ab2"` pattern: a τ-1 *tendency* carry, separate `dt_tracer`/`dt_mom`, and the implicit vertical mixing applied ONCE per step (NOT doubled). **A leapfrog+Robert-Asselin attempt (gates I1+I2) was built, measured + reviewed, then REVERTED (commit `d1648f8e`)** — it was (a) the wrong scheme and (b) UNSTABLE: the extract-from-FE wrapper (`X^{n+1}=X^{n-1}+2(X_FE-X^n)`) could not separate advection from diffusion, so it leapfrogged the implicit vertical mixing (doubling the backward-Euler increment, `2·Δ(dt)` vs `Δ(2dt)`) → the free-run blew up at 3 days; the 40-step unit test masked a 230× computational-mode growth. The measure-first + adversarial-review process caught the misdirected build before it shipped. | **AB2 BUILT (gate A1) — but NOT the 30-day lever.** `outer_integrator="ab2"` (extract-from-FE: AB2 the explicit FE increment `X^{n+1}=X^n+(1.5+ε)·ΔX^n−(0.5+ε)·ΔX^{n-1}`; the barotropic mode is kept from the split-explicit free-surface solve, un-AB2'd; the baroclinic momentum deviation is AB2'd). **STABLE** — 8 unit tests + a 300-step channel + the real-ACC **30-day free-run** (where the leapfrog blew up at 3 days); default `forward_euler` bit-identical (36 tests). AB2 applies the implicit-mixing increment ~1× (not the leapfrog's fatal 2×). **Measure-first: the gap is UNCHANGED (+70.3%/+233.4%, same as forward-Euler)** ⇒ the integrator SCHEME (FE vs AB2) is NOT the 30-day lever. CAVEAT (adversarial-review #2): this AB2s the TOTAL increment (incl. the once-applied implicit vertical mixing) rather than Veros's faithful explicit-AB2 + implicit-once, and uses a single dt (not `dt_tracer≠dt_mom`). Consequence: vertical-mixing stability is CONDITIONAL (threshold `dt·K_v·4/dz²_min ≲ a few`; a stiff-K_v channel blows up at ~200 steps, while forward-Euler — implicit-once — is unconditionally stable). Safe for mild mixing (ACC's TKE: `dt·λ≈-0.2`); `step` GUARDS against `outer_integrator="ab2"` + convective adjustment, and a stiff-K_v test locks the regression. Full fidelity (explicit/implicit split + dt split) is the documented refinement. **NEXT lever (NOT the integrator scheme): the energetics.** A **1-year free-run** shows the gap is PERSISTENT, not spin-up — transport +58% (was +70% @30d), total KE **+443%** (was +233%), max|u| **+124%** (was −14%): legoESM's ACC becomes ~5× more energetic than Veros while the thermal mean state matches (vol-mean T +0.5%). The EKE/GM eddy damping is still cold at 1 year (multi-year equilibration) in BOTH models, so it isn't yet arresting the jet. Candidates: (a) the **barotropic formulation** (Veros rigid-lid/streamfunction `solve_stream.py` vs legoESM split-explicit free surface — but matching the rigid-lid would REGRESS legoESM's modern free surface, so likely an accept-the-delta with justification, not a must-build); (b) an **eddy/momentum dissipation deficit** (legoESM accumulating KE faster — bottom drag / lateral viscosity / barotropic-mode damping); (c) the `dt_tracer≠dt_mom` split. A barotropic-vs-baroclinic KE decomposition (where the excess energy lives) is the bounded next diagnostic. **DISSIPATION AUDIT (2026-05-29) — found a bottom-drag mis-mapping AND, by fixing it, isolated the genuine deeper difference: the BAROTROPIC FORMULATION.** (1) The recipe copied Veros's `r_bot=1e-5` into `bottom_drag_r=1e-5`, but Veros applies it as a RATE on the bottom cell (`du/dt=-r_bot·u`, NO ÷dz; `friction.py:284-289`) while legoESM uses the stress form `du/dt=-r·u/h_bot` — so the identical value made legoESM's drag ~`h_bot`≈276× TOO WEAK. The FAITHFUL mapping is `R_BOT=r_bot·h_bot≈2.76e-3` (adopted). (2) BUT the faithful drag does NOT reconcile the transport — it OVER-damps it: @30d it looked like a clean fix (KE +233%→+27%, transport +70%→−22%), but **@1yr the faithful drag gives transport −83% (18 vs 110 Sv) / KE −29%**, vs the mis-mapped `r=1e-5`'s +58% / +443%. So legoESM's barotropic transport OVER-RESPONDS to bottom drag vs Veros (same bottom-cell rate → 18 Sv legoESM vs 110 Sv Veros) ⇒ the genuine deeper difference is the **barotropic formulation** (legoESM split-explicit free surface vs Veros rigid-lid/streamfunction); the old `r=1e-5` had masked it via COMPENSATING errors (under-drag ≈ cancelled the over-responsive barotropic). We keep the FAITHFUL drag (match Veros's scheme, not tune to the metric); the large transport residual is the barotropic-formulation delta. **The barotropic formulation (free-surface vs rigid-lid) is the clearly-isolated remaining ACC gap.** **RIGID-LID BUILT + VALIDATED (2026-05-29) — hypothesis CONFIRMED.** A full general rigid-lid streamfunction solver was added as `barotropic_solver="rigid_lid"` (faithful port of Veros `solve_stream.py`: the weighted vertex Laplacian `∇·((1/H)∇)ψ = curl_vertex(velocity_recovery(ψ))` reusing the audited C-grid stencils; AD-safe symmetric-CG elliptic solve (on the SPD form `−A_vertex·L`, Jacobi-preconditioned — adversarial review caught a BiCG-STAB adjoint-VJP NaN that broke `jax.grad`, and CG's symmetric `custom_linear_solve` adjoint fixed it; gradient now matches finite-difference); the multiply-connected island machinery — flood-fill, `psin` basis, `line_psin` coupling — with the channel-transport line integral via discrete Stokes; AB2 ψ-stepping; Coriolis added to the streamfunction RHS). **1-year ACC: transport O(100 Sv) — 78–115 Sv across solver round-off, chaotically variable at 1 yr (see below) — versus the free-surface's collapsed 18 Sv (−83%) at the same faithful bottom drag** (Veros 110.6 Sv). The rigid lid RESTORES a Veros-magnitude ACC, *confirming* that the barotropic formulation (free-surface over-responding to bottom drag) was the dominant control on the ACC transport. NB the precise value is NOT pinned: the two solvers (the original BiCG-STAB and the shipped CG, see below) agree bit-for-bit through the 30-day spinup (both 150.6 Sv) and diverge only as turbulence develops — the flow is over-energetic (KE ≈19× Veros) so instantaneous transport varies chaotically; a tight quantitative match needs the dissipation fix below + a time-mean transport, not a 1-yr snapshot. SECONDARY (KE excess — INVESTIGATED 2026-05-29, prompted by "are the dissipation schemes actually matched?"). The rigid-lid KE looked ≈19× Veros at 1 yr. Audit: the dissipation **coefficients MATCH** (A_h=2.2e5 harmonic cos¹, centered advection, B_h=0, bottom drag, GM/Redi, TKE). The "19×" turned out to be (a) PARTLY a **premature 1-yr comparison** — Veros's KE *also* climbs all year (3.1e15@30d → 8.6e15@90d → 1.4e16@180d → 2.1e16@365d, never plateauing; both models are pre-equilibrium under the cold prognostic-EKE/GM sink), so the equilibrium gap is overstated; and (b) a GENUINE but milder over-energisation — legoESM is 4.6×@30d → 8.5×@90d → 19×@365d, i.e. more energetic *early* (before eddies dominate) and *widening*, tracking the ACC **spin-up overshoot** (legoESM hits 150 Sv by day 30 vs Veros ~35; the developing eddies then amplify the gap). It is **NOT a numerics bug**: the operator has no checkerboard nullmode (cb has 127× larger response than a smooth mode); the momentum-advection scheme is not the source (flux-form and the energy-conserving vector-invariant grow identically); and free-decay tests show the stratification FLATTENS as KE grows (⇒ physical baroclinic APE→KE, not a spurious source). My earlier "missing barotropic dissipation / add a damper" was WRONG (tune-to-metric; the free surface's lower KE was its *numerical* substep filter + a weak 18 Sv ACC, not validated dissipation). **ENERGY BUDGET (both models, 2026-05-29) — root cause localized.** Built a KE + APE(=PE−RPE, sorted, rigorous) + wind-power diagnostic and ran legoESM rigid-lid vs bridged Veros at 30/90/180 d. (1) **Wind RULED OUT** — P_wind is comparable, Veros even higher (1.65e11 vs 1.11e11 W @180d); legoESM is more energetic *despite less* wind (corrects the "spin-up→more wind" guess). (2) Energy is **buoyancy-driven, not wind** (d(KE+APE)/dt ≈ 2.3e12 W ≫ P_wind ≈ 1.1e11 W). (3) Energy is **APE-dominated** (APE/KE ≈ 30–100) and **legoESM holds ≈5× more APE than Veros** (2.0e19 vs 4.1e18 J @180 d). **The restoring is NOT the differentiator** (verified, user-prompted): the restoring *scheme* is identical and its measured APE generation G is comparable — Veros's is even *higher* (5.95e11 vs 2.87e11 W @180 d) — yet Veros holds 5× *less* APE. So the differentiator is the **APE drainage, not the generation**: Veros drains its APE effectively, legoESM doesn't. ⇒ legoESM runs hot because its **APE isn't drained**: the sink is the **GM eddy-induced isopycnal flattening**, weak under the *cold-started prognostic EKE* (the mean front is bounded by the same T\* in both — the 5× is *eddy* APE governed by the GM sink), amplified by legoESM's stronger ACC. (Caveat: the *absolute* budget doesn't fully close — legoESM's d(KE+APE)/dt exceeds wind + the surface-only G probe; the *comparison* is robust, full closure needs the explicit conversion + dissipation terms.) **ADIABATIC CONSERVATION TESTS (closed-budget certainty).** (A) Uniform-T/S adiabatic core (no forcing/dissipation/stratification): **KE is conserved** for both advection schemes (~92%, flat) — no spurious momentum/advection source (the vector-invariant advection is a *tested* Arakawa-Lamb-81 energy+enstrophy-conserving scheme). (B) **RPE flat to 6 digits** adiabatically — no spurious mixing. (C) the adiabatic KE growth is ~4% of the forced KE growth ⇒ **the KE excess is ~96% physical, not a spurious KE source.** ⇒ **HIGH-CONFIDENCE: the KE excess is the GM/EKE eddy-sink spin-up regime (physics), not a numerical artifact.** ONE bounded loose end: the *stratified* adiabatic run doesn't conserve **KE+PE** (PE grows; in-situ pressure doesn't fix it) — *either* a real **Adcroft-PGF energy non-conservation** (the PGF is built for partial-cell accuracy, not designed/tested EC — a *general* dycore property, same PGF in free-surface runs) *or* my continuous PE ≠ the discrete-PGF-consistent PE. Minor for KE; closing it needs the discrete-Adcroft-consistent APE — a focused dycore follow-up, decoupled from the ACC result. **ZONAL MOMENTUM BUDGET (both models, robust — clean state integrals, no APE/PGF issues).** Sanity: wind input 3.16e12 N identical (flow-independent), Coriolis ~0 (mass conservation). (1) **Flat-bottom Munk balance** in both: bottom drag is only ~10–30% of the wind; ~70% is removed by the **lateral wall viscous stress** (the large Munk A_h=2.2e5 absorbs the wind at the side walls — no form stress on a flat bottom). (2) **Vertical structure differs:** legoESM's ACC is **more barotropic** (u_bot/u_surf ≈ 0.3–0.5) vs Veros's **surface-intensified** (≈ 0.13–0.22) — legoESM's wind momentum penetrates deeper ⇒ higher bottom velocity ⇒ more bottom drag + more total momentum. This is the **same story as the energy budget from an independent diagnostic**: legoESM's more-energetic eddies (weak GM under cold EKE) transfer wind momentum downward (interfacial form stress) ⇒ a more barotropic ACC. Distinct from the barotropic *formulation* (the rigid-lid matched the transport *magnitude*); this is the vertical *distribution*, set by the eddy field. An eddy-parameterisation spin-up/equilibration regime — NOT wind, NOT numerics (schemes match, physical APE→KE). The diagnostic is a script; promote to a tested `budgets.py` module + add the explicit APE→KE conversion term for a permanent closed budget if wanted. 13 direct unit tests; full free-surface path bit-unchanged (default solver). | **RIGID-LID BUILT + VALIDATED: restores an O(100 Sv) ACC (free-surface collapsed to 18 Sv) ⇒ barotropic-formulation hypothesis CONFIRMED; gradient-NaN fixed (symmetric CG, adversarial review). KE excess INVESTIGATED: dissipation schemes MATCH; the "19×@1yr" is partly a premature comparison (Veros KE also climbs all year, both pre-equilibrium under cold GM) + a genuine milder over-energisation (4.6×@30d→19×@365d). ENERGY BUDGET (both models) localized the root: wind RULED OUT (Veros's P_wind is higher); energy is buoyancy-driven (restoring→APE); legoESM builds ≈5× more APE ⇒ its APE isn't drained fast enough = the GM eddy-flattening sink set by the cold prognostic EKE (front oversteepens), amplified by the stronger ACC. NOT a numerics bug, NOT missing dissipation, NOT the wind. Transport result stands.** |
./docs/ocean/fidelity/oracle_recipe_strategy.md-544-| **Veros isopycnal/GM-Redi discretization** — we have GM/Redi (`slope_scheme` = triads/centered); Veros's isoneutral scheme differs (T_iso corr 0.17). | yes (have variant) | Missing **variant** | **DONE (root-caused + built, `31313850`): Veros-faithful NEUTRAL-density isoneutral slopes.** legoESM built slopes from IN-SITU density, whose vertical gradient carries compressibility (∂ρ/∂p·∂p/∂z ≈ 4.5e-3 kg/m⁴) ⇒ \|∂_zρ\| ~4× too steep ⇒ slopes 4× too flat ⇒ K_33 ∝ S² collapsed 10–25× interior / 10⁴× near-surface. Veros uses locally-referenced NEUTRAL gradients `drdT·∇T + drdS·∇S` (`isoneutral.py:40-41`). Added `eos.eos_density_derivatives` (jax.grad of any EOS at fixed p) + `GMRediConfig.slope_density="in_situ"\|"neutral"` (default in_situ BIT-IDENTICAL; ACC recipe opts in). KEY: Veros never CLIPS the slope (DM95 taper only) — the neutral path is unclipped (clipping would saturate the taper and inflate K_33 ~10×). Truth tiers hold (bit-identity Δ=0, Redi cancellation 1e-17, conservation 1e-17, AD=FD). **T_iso 0.17→0.37 (Veros-K_iso-fed = the meaningful metric); K_33 now ~2× of Veros vs 20–100× too small.** Same in-situ-vs-locally-referenced class as the convection-N² fix (`dfc8318b`). NB the constant-kappa frozen-state probe reads 0.077 — a KAPPA confound (constant 1000 ≠ Veros's variable K_iso), not a regression. **kr-sum refinement DONE (`37112b46`, 2026-06-09):** every triad now pairs its reference cell's EOS derivatives in numerator AND denominator (W-face B-triads + u/v-face above-faces; Veros kr/ki loops, term-for-term 0.0). Measured inert on the uniform ACC (T_iso 0.3665→0.3666, as predicted — bites at variable metrics/tripolar) but it FIXED a real adiabaticity defect: per-triad Redi cancellation 3.5e-11 → 8.9e-26 (the above-face triads had mixed cells between numerator and denominator). Remaining (low): exact per-corner metric factors (dxu/dxt·cost weights — inert on uniform grids); the `min(x,−eps)` vs Veros `min(0,x)−eps` floor = equivalent regularization (~1e-6 interior), documented. | **DONE** |
./docs/ocean/fidelity/oracle_recipe_strategy.md-545-| **IDEMIX** — internal-wave energy/mixing. | **no** (`enable_idemix=False`) | n/a | **Defer** — Veros runs ACC without it, so neither model needs it for apples-to-apples. | — |
./docs/ocean/fidelity/oracle_recipe_strategy.md-546-| **Implicit vs explicit vertical mixing** (`du_mix` corr ~0). | implicit | — (config exists) | **Accept** — both paths exist; recipe selects implicit. Not a gap. | — |
./docs/ocean/fidelity/oracle_recipe_strategy.md-547-| **Density / EOS / hydrostatic pressure** (corr 1.0). | yes | — | **Accept** — already matches (shared discretization). | — |
./docs/ocean/fidelity/oracle_recipe_strategy.md-548-
--
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
--
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
--
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
--
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
--
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
--
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
--
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
--
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
--
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
--
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
--
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
--
scripts/bench/metadata.py-621-        "sypd": sypd,
scripts/bench/metadata.py-622-        "mcells_per_s": mcells_per_s,
scripts/bench/metadata.py-623-    }
scripts/bench/metadata.py-624-
scripts/bench/metadata.py-625-
scripts/bench/metadata.py:626:#: Placeholder comm-fabric numbers for :func:`calibrated_bound` when the
scripts/bench/metadata.py-627-#: caller passes no measured values.  Ballpark single-node GPU-interconnect
scripts/bench/metadata.py-628-#: figures (order NVLink/PCIe), NOT measurements of THIS machine —
scripts/bench/metadata.py-629-#: MACHINE-CALIBRATED-REQUIRED: any bound built on them is emitted with
scripts/bench/metadata.py-630-#: ``bound_calibrated=False`` and must never be quoted as a hardware
scripts/bench/metadata.py-631-#: roofline.  Calibrate with a ping-pong / allreduce microbenchmark on the
--
scripts/bench/metadata.py-749-        out["wet_cell_levels_per_device_min"] = min(per)
scripts/bench/metadata.py-750-        out["wet_cell_levels_per_device_max"] = max(per)
scripts/bench/metadata.py-751-    return out
scripts/bench/metadata.py-752-
scripts/bench/metadata.py-753-
scripts/bench/metadata.py:754:def calibrated_bound(
scripts/bench/metadata.py-755-    *,
scripts/bench/metadata.py-756-    measured_fused_step_ms: float | None = None,
scripts/bench/metadata.py-757-    single_device_fused_step_ms: float | None = None,
scripts/bench/metadata.py-758-    halo_messages_per_step: int | None = None,
scripts/bench/metadata.py-759-    halo_bytes_per_step: int | None = None,
--
scripts/bench/metadata.py-811-    lat_us = (DEFAULT_COMM_LATENCY_US if latency_us is None
scripts/bench/metadata.py-812-              else float(latency_us))
scripts/bench/metadata.py-813-    bw_gbs = (DEFAULT_COMM_BANDWIDTH_GBS if bandwidth_GBs is None
scripts/bench/metadata.py-814-              else float(bandwidth_GBs))
scripts/bench/metadata.py-815-    if lat_us < 0.0:
scripts/bench/metadata.py:816:        raise ValueError(f"calibrated_bound: latency_us must be >= 0, "
scripts/bench/metadata.py-817-                         f"got {lat_us}")
scripts/bench/metadata.py-818-    if bw_gbs <= 0.0:
scripts/bench/metadata.py:819:        raise ValueError(f"calibrated_bound: bandwidth_GBs must be > 0, "
scripts/bench/metadata.py-820-                         f"got {bw_gbs}")
scripts/bench/metadata.py-821-
scripts/bench/metadata.py-822-    missing = [name for name, v in (
scripts/bench/metadata.py-823-        ("single_device_fused_step_ms", single_device_fused_step_ms),
scripts/bench/metadata.py-824-        ("halo_messages_per_step", halo_messages_per_step),
--
./tests/bench/test_bench_cube_tiled_step_scaling.py-54-        mod.main()
./tests/bench/test_bench_cube_tiled_step_scaling.py-55-
./tests/bench/test_bench_cube_tiled_step_scaling.py-56-
./tests/bench/test_bench_cube_tiled_step_scaling.py-57-def test_collective_census_helper():
./tests/bench/test_bench_cube_tiled_step_scaling.py-58-    hlo = "\n".join([
./tests/bench/test_bench_cube_tiled_step_scaling.py:59:        "%x = collective-permute(...)",
./tests/bench/test_bench_cube_tiled_step_scaling.py:60:        "%y = collective-permute-start(...)",
./tests/bench/test_bench_cube_tiled_step_scaling.py:61:        "%z = collective-permute-done(...)",  # not counted (done)
./tests/bench/test_bench_cube_tiled_step_scaling.py-62-        "%w = add(...)",
./tests/bench/test_bench_cube_tiled_step_scaling.py-63-    ])
./tests/bench/test_bench_cube_tiled_step_scaling.py-64-    assert mod._count_collective_permutes(hlo) == 2
./tests/bench/test_bench_cube_tiled_step_scaling.py-65-
./tests/bench/test_bench_cube_tiled_step_scaling.py-66-
--
scripts/bench/bench_ocean_latlon_spmd_pcg.py-11-mpi4jax — runs on route-B (cuda-jax, the RTX8000 PCIe pair) where mpi4jax
scripts/bench/bench_ocean_latlon_spmd_pcg.py-12-is unavailable.
scripts/bench/bench_ocean_latlon_spmd_pcg.py-13-
scripts/bench/bench_ocean_latlon_spmd_pcg.py-14-STRONG scaling: a FIXED global grid solved on 1 device (whole grid) then N
scripts/bench/bench_ocean_latlon_spmd_pcg.py-15-devices (each a latitude band).  ``efficiency(N) = t(1) / (N * t(N))``;
scripts/bench/bench_ocean_latlon_spmd_pcg.py:16:ideal 1.0.  On a PCIe pair (no NVLink) the psum collective-permute is the
scripts/bench/bench_ocean_latlon_spmd_pcg.py-17-expected ceiling — this bench MEASURES that ceiling.
scripts/bench/bench_ocean_latlon_spmd_pcg.py-18-
scripts/bench/bench_ocean_latlon_spmd_pcg.py-19-The operator is the uniform-coefficient 5-point Helmholtz ``A = I +
scripts/bench/bench_ocean_latlon_spmd_pcg.py-20-c(-Delta)`` (built on the backend-oblivious ``pad_halo_latlon``, so the
scripts/bench/bench_ocean_latlon_spmd_pcg.py-21-SAME closure runs serial and sharded).  Its COMMUNICATION pattern (band
--
scripts/bench/run_scaling_diagnosis.py-746-            if c is None:
scripts/bench/run_scaling_diagnosis.py-747-                print("  census: unavailable (step not lowerable on this "
scripts/bench/run_scaling_diagnosis.py-748-                      "backend)")
scripts/bench/run_scaling_diagnosis.py-749-            else:
scripts/bench/run_scaling_diagnosis.py-750-                print(f"  {census_info['n_devices']} device(s): "
scripts/bench/run_scaling_diagnosis.py:751:                      f"{c['collective_permute']} collective-permute, "
scripts/bench/run_scaling_diagnosis.py-752-                      f"{c['all_reduce']} all-reduce, "
scripts/bench/run_scaling_diagnosis.py-753-                      f"{c['all_gather']} all-gather / step")
scripts/bench/run_scaling_diagnosis.py-754-                if census_info["n_devices"] == 1:
scripts/bench/run_scaling_diagnosis.py-755-                    print("  (single device: no inter-device schedule — run "
scripts/bench/run_scaling_diagnosis.py-756-                          "with XLA_FLAGS=--xla_force_host_platform_device_"
--
./docs/ocean/fidelity/dino_wiring_diagram.md-132-  a separate Matsuno rotation — possible double-count if Matsuno rotates the FULL u
./docs/ocean/fidelity/dino_wiring_diagram.md-133-  (not u') and Phase-3 doesn't overwrite it. JET-RELEVANT — deeper trace queued.
./docs/ocean/fidelity/dino_wiring_diagram.md-134-
./docs/ocean/fidelity/dino_wiring_diagram.md-135-**Node 8/16/17 Coriolis-split — VERIFIED (no double-count).** An agent claimed the
./docs/ocean/fidelity/dino_wiring_diagram.md-136-barotropic Coriolis is double-counted under the default `matsuno_split`, but code
./docs/ocean/fidelity/dino_wiring_diagram.md:137:inspection REFUTES it (4th agent over-claim this loop): under `matsuno_split`, `du_dt`
./docs/ocean/fidelity/dino_wiring_diagram.md-138-EXCLUDES the planetary Coriolis (the stage-7b' planetary-Coriolis add,
./docs/ocean/fidelity/dino_wiring_diagram.md-139-`ocean_pe_latlon_cgrid.py:3627`, is gated on `coriolis_scheme=="explicit_ab2"`), so
./docs/ocean/fidelity/dino_wiring_diagram.md-140-`F_slow` carries NO f, and the substep applies `f×U_bt` exactly once (`_add_bt_cor=True`,
./docs/ocean/fidelity/dino_wiring_diagram.md-141-`ocean_model_latlon_cgrid.py:2544-2546`); the Matsuno rotation applies f to the
./docs/ocean/fidelity/dino_wiring_diagram.md-142-perturbation. Under `explicit_ab2`, `du_dt` HAS f → `F_slow` carries it → substep skips
--
./docs/ocean/fidelity/dino_wiring_diagram.md-495-   AB2** explicit Coriolis does NOT → the mode grows. z* has NO staircase ⇒ no HPG error ⇒ no
./docs/ocean/fidelity/dino_wiring_diagram.md-496-   source (why z*+explicit is stable). The leapfrog blows EARLIER than AB2 (step 22 vs 26)
./docs/ocean/fidelity/dino_wiring_diagram.md-497-   because rDt=2dt takes larger steps of the same growing physical mode — consistent with a
./docs/ocean/fidelity/dino_wiring_diagram.md-498-   physical (not computational-mode) instability, so the Asselin filter does not arrest it.
./docs/ocean/fidelity/dino_wiring_diagram.md-499-
./docs/ocean/fidelity/dino_wiring_diagram.md:500:**Faithful lever CONFIRMED but insufficient alone: node 14 (lateral viscosity high-lat
./docs/ocean/fidelity/dino_wiring_diagram.md-501-scaling).** `nemo_dino_kamm` runs `A_h_floor=0, A_h_eq_boost=1` (NEMO-faithful, no legoESM
./docs/ocean/fidelity/dino_wiring_diagram.md-502-stabiliser) so A_h·cosφ drops to ~34% at 70°. NEMO's `ahmf = ½·rn_Uv·max(e1,e2)` does NOT
./docs/ocean/fidelity/dino_wiring_diagram.md-503-shrink at high lat (e2=R·Δφ dominates) ⇒ **NEMO's high-lat viscosity is ~3× legoESM's** — a
./docs/ocean/fidelity/dino_wiring_diagram.md-504-genuine node-14 mismatch, not a stabiliser. Removing the cosφ reduction (≈node 14) **delays the
./docs/ocean/fidelity/dino_wiring_diagram.md-505-blow-up step 26→34 and halves the growth rate**, but does NOT close it. So node 14 is a real
--
./docs/ocean/fidelity/eke_build_progress.md-155-Ran a developed-flow Veros ACC integration (enable_eke; runlen 864000 s = 10 d) and captured
./docs/ocean/fidelity/eke_build_progress.md-156-Veros's 3-D eke / K_gm / eke_len. KEY FINDING (verified from veros/core/eke.py + acc.py):
./docs/ocean/fidelity/eke_build_progress.md-157-  Veros K_gm = min(eke_k_max, eke_c_k·eke_len·√eke)  — IDENTICAL FORM to legoESM eke_kappa_gm
./docs/ocean/fidelity/eke_build_progress.md-158-  Veros dissipation c_int = eke_c_eps·√eke/eke_len   — IDENTICAL to legoESM eps = c_eps·E^{3/2}/L
./docs/ocean/fidelity/eke_build_progress.md-159-The ACC setup sets eke_c_k=0.4, eke_c_eps=0.5, eke_k_max=1e4, eke_lmin=100, superbee advection,
./docs/ocean/fidelity/eke_build_progress.md:160:isopycnal diffusion — ALL match EKEConfig() defaults exactly. NUMERICAL CONFIRMATION: feeding
./docs/ocean/fidelity/eke_build_progress.md-161-Veros's OWN eke + eke_len into legoESM eke_kappa_gm reproduces Veros's K_gm to MACHINE PRECISION
./docs/ocean/fidelity/eke_build_progress.md-162-at the matching time level (tau=2: max_rel_err=0.0, corr=1.000000, 19560 wet cells). Reproducible:
./docs/ocean/fidelity/eke_build_progress.md-163-scripts/validate/ocean_fidelity/compare_eke_kappa_veros.py.
./docs/ocean/fidelity/eke_build_progress.md-164-GAP SURFACED (the next must-build, NOT part of E1-E9): the MIXING LENGTH differs.
./docs/ocean/fidelity/eke_build_progress.md-165-  Veros eke_len = max(eke_lmin, min(eke_cross·L_rossby, eke_crhin·L_rhines)), eke_cross=2, with the
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-56-# under rec["metadata"] so a virtual-CPU-device proxy, a gloo/TCP fabric run,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-57-# or an f32 ablation is falsifiable from the JSONL row alone.  metadata.py
scripts/bench/bench_ocean_latlon_spmd_scaling.py-58-# imports JAX lazily, so this is safe before jax.distributed.initialize.
scripts/bench/bench_ocean_latlon_spmd_scaling.py-59-from metadata import (  # noqa: E402
scripts/bench/bench_ocean_latlon_spmd_scaling.py-60-    annotate_incomplete,
scripts/bench/bench_ocean_latlon_spmd_scaling.py:61:    calibrated_bound,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-62-    comm_accounting,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-63-    scaling_metadata,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-64-    tidy_throughput_fields,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-65-    wet_cell_metrics,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-66-)
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-286-    p.add_argument("--fused-halo", action="store_true",
scripts/bench/bench_ocean_latlon_spmd_scaling.py-287-                   help="Opt-in SPMD halo message aggregation "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-288-                        "(LEGOESM_LATLON_SPMD_FUSED_HALO=1): one ppermute "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-289-                        "pair per direction per dtype group at every "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-290-                        "pad_multi site instead of one per field — "
scripts/bench/bench_ocean_latlon_spmd_scaling.py:291:                        "measured 25%% fewer static collective-permutes on "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-292-                        "this step, bit-identical results. A/B against "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-293-                        "the default run.")
scripts/bench/bench_ocean_latlon_spmd_scaling.py-294-    p.add_argument("--wide-halo", action="store_true",
scripts/bench/bench_ocean_latlon_spmd_scaling.py-295-                   help="Opt-in wide-halo split-explicit barotropic: one "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-296-                        "fused wide lat-halo exchange per chunk of substeps "
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-605-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-606-    # --- Communication accounting (audit item 4) + calibrated bound (8) ----
scripts/bench/bench_ocean_latlon_spmd_scaling.py-607-    # Analytic INTER-DEVICE census, barotropic-solver scope ONLY (the
scripts/bench/bench_ocean_latlon_spmd_scaling.py-608-    # baroclinic 3-D pads are not counted -> bytes/comm are a LOWER census,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-609-    # flagged machine-readably via halo_bytes_is_lower_bound; T_bound is a
scripts/bench/bench_ocean_latlon_spmd_scaling.py:610:    # heuristic model, see calibrated_bound's docstring).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-611-    # implicit_cn PCG: each Helmholtz apply pads eta N+S (gradient stencil)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-612-    # + the v-face flux row (divergence) ~= 2 exchanges/apply, applied
scripts/bench/bench_ocean_latlon_spmd_scaling.py-613-    # iters + 1 times (incl. the initial residual); reductions = the dot
scripts/bench/bench_ocean_latlon_spmd_scaling.py-614-    # batches (2/iter standard, 1/iter single_reduce) + the initial batch
scripts/bench/bench_ocean_latlon_spmd_scaling.py-615-    # + the mass-projection psum + the eta-floor clamp psum.
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-677-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-678-    # Calibrated T_bound (audit item 8): nd=1 rows ARE their own compute
scripts/bench/bench_ocean_latlon_spmd_scaling.py-679-    # ingredient; nd>1 rows need the nd=1 fused number passed in (else the
scripts/bench/bench_ocean_latlon_spmd_scaling.py-680-    # bound is emitted null + flagged).  launch_host_ms=0: per-step launch
scripts/bench/bench_ocean_latlon_spmd_scaling.py-681-    # cost inside a fused lax.scan block is amortized to ~0.
scripts/bench/bench_ocean_latlon_spmd_scaling.py:682:    bound_rec = calibrated_bound(
scripts/bench/bench_ocean_latlon_spmd_scaling.py-683-        measured_fused_step_ms=med,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-684-        single_device_fused_step_ms=(med if nd == 1
scripts/bench/bench_ocean_latlon_spmd_scaling.py-685-                                     else args.single_dev_fused_ms),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-686-        halo_messages_per_step=comm_rec["halo_messages_per_step"],
scripts/bench/bench_ocean_latlon_spmd_scaling.py-687-        halo_bytes_per_step=comm_rec["halo_bytes_per_step"],
--
./docs/ocean/fidelity/nemo_gyre_fidelity_plan.md-239-  mixed state.** Summer subtropical avt NEVER drops to background — it stays
./docs/ocean/fidelity/nemo_gyre_fidelity_plan.md-240-  0.03-0.09 (3000-6000x the 1.2e-5 floor) down to ~380 m ALL YEAR. Depth
./docs/ocean/fidelity/nemo_gyre_fidelity_plan.md-241-  profile at day152 (summer): N²≈+4e-10 (essentially NEUTRAL, T uniform
./docs/ocean/fidelity/nemo_gyre_fidelity_plan.md-242-  17.36), TKE energy e=1.5-2.9e-3 (≈1000x the ~1e-6 floor) SUSTAINED to
./docs/ocean/fidelity/nemo_gyre_fidelity_plan.md-243-  234 m, mixing length l_k grows to ~200 m (buoyancy length √(2e)/N blows up
./docs/ocean/fidelity/nemo_gyre_fidelity_plan.md:244:  as N→0, capped only by distance-to-surface). Feedback: N²≈0 → l~200 m →
./docs/ocean/fidelity/nemo_gyre_fidelity_plan.md-245-  avt large → column stays neutral → l~200 m. The first winter (d30) is still
./docs/ocean/fidelity/nemo_gyre_fidelity_plan.md-246-  STRATIFIED (N²=+6.3e-5, e=0 at depth, thermocline intact) — the lock closes
./docs/ocean/fidelity/nemo_gyre_fidelity_plan.md-247-  in spring (d30 contrast 0.22 → d61 0.00). Each summer surface heating tries
./docs/ocean/fidelity/nemo_gyre_fidelity_plan.md-248-  to restratify the top but the sustained mixing homogenises it as fast as it
./docs/ocean/fidelity/nemo_gyre_fidelity_plan.md-249-  is applied → mixing wins → thermocline never rebuilds.
--
./docs/ocean/fidelity/autonomous_progress.md-24-`build_region_masks` include the zero-padded wall cells (so walls explain interior L2=5.1),
./docs/ocean/fidelity/autonomous_progress.md-25-or is interior already wall-clean (so the residual is cause (a) time-level, not walls)?
./docs/ocean/fidelity/autonomous_progress.md-26-Action this iter: read `tendency_probe.py` (build_region_masks + rho comparison) and
./docs/ocean/fidelity/autonomous_progress.md-27-`veros_state_bridge.py` (wall-row padding + land_mask handling) to settle it.
./docs/ocean/fidelity/autonomous_progress.md-28-
./docs/ocean/fidelity/autonomous_progress.md:29:### 2026-05-28 · iter 1 · DIAGNOSIS CONFIRMED + FIX APPLIED
./docs/ocean/fidelity/autonomous_progress.md-30-Confirmed (CPU probe): ACC grid is 44 rows = NY(42) + 2 padding rows at lat=-41 and +45,
./docs/ocean/fidelity/autonomous_progress.md-31-OUTSIDE the physical domain [-40,+44]. Both padding rows are WET in the recipe land_mask
./docs/ocean/fidelity/autonomous_progress.md-32-(row0 100% via lat<-20; row-1 97% via lon>1) and land in `interior` (435+405=840 cells).
./docs/ocean/fidelity/autonomous_progress.md-33-The bridge (`veros_state_bridge._pad_y_walls`, lines 201-208) zero-fills their T,S → EOS
./docs/ocean/fidelity/autonomous_progress.md-34-sees T=S=0 → rho≈997 → −27 kg/m³/cell. Predicted interior L2 ≈ 5.79 kg/m³ vs observed 5.1
./docs/ocean/fidelity/autonomous_progress.md:35:→ CAUSE CONFIRMED = non-physical zero-padded rows in the comparison, NOT a numerics/EOS bug.
./docs/ocean/fidelity/autonomous_progress.md-36-FIX: `build_acc_land_mask` now marks rows with lat outside [Y_ORIGIN, Y_ORIGIN+NY*DYT] as
./docs/ocean/fidelity/autonomous_progress.md-37-land (in-construction via land_mask_override → u/v masks stay consistent). No production
./docs/ocean/fidelity/autonomous_progress.md-38-numerics touched.
./docs/ocean/fidelity/autonomous_progress.md-39-GATE (a) [no wet T=S=0 cells]: GREEN — new test `test_bridged_acc_state_has_no_zero_TS_wet_cells`
./docs/ocean/fidelity/autonomous_progress.md-40-+ `test_acc_land_mask_marks_out_of_domain_padding_rows_as_land`.
--
./docs/ocean/fidelity/autonomous_progress.md-288-
./docs/ocean/fidelity/autonomous_progress.md-289-### 2026-05-28 · iter 16 · A_h check (user-directed) surfaced + fixed a real recipe GM/Redi bug
./docs/ocean/fidelity/autonomous_progress.md-290-User picked "dynamics A_h canonical" and said "make sure you check it." The check (2 Explore
./docs/ocean/fidelity/autonomous_progress.md-291-agents DISAGREED -> resolved by direct reads) found:
./docs/ocean/fidelity/autonomous_progress.md-292-- A_h: config.A_h (dynamics, ocean_pe_latlon_cgrid §10) IS the lat-lon viscosity — recommendation
./docs/ocean/fidelity/autonomous_progress.md:293:  CONFIRMED. The physics-pathway lateral mixing is cubed-sphere-only (harmonic/biharmonic crash
./docs/ocean/fidelity/autonomous_progress.md-294-  on lat-lon shapes; gm_redi raises TypeError) — so it's a cryptic-crash footgun, not silent
./docs/ocean/fidelity/autonomous_progress.md-295-  double-application.
./docs/ocean/fidelity/autonomous_progress.md-296-- BIGGER (incidental): the lat-lon MODEL applies GM/Redi from the TOP-LEVEL config.gm_redi
./docs/ocean/fidelity/autonomous_progress.md-297-  (ocean_model_latlon_cgrid.py:998, default None), but the recipe set GM/Redi ONLY in
./docs/ocean/fidelity/autonomous_progress.md-298-  physics.lateral_mixing.gm_redi -> the model ignored it -> **the ACC recipe's GM/Redi was
--
./docs/ocean/fidelity/autonomous_progress.md-384-  not by field position; fixed stale "~45 fields"->"~70".
./docs/ocean/fidelity/autonomous_progress.md-385-- New `tests/ocean/unit/test_config_footguns.py` (7 tests, all green). 35 related tests green.
./docs/ocean/fidelity/autonomous_progress.md-386-- **Pre-existing, unrelated failure noted (NOT a regression):** test_freshwater.py::
./docs/ocean/fidelity/autonomous_progress.md-387-  TestCouplerAdapter::test_compute_mpas_freshwater_basic fails with `SurfaceToAtm.__new__() got an
./docs/ocean/fidelity/autonomous_progress.md-388-  unexpected keyword argument 'T_water_init_C'` — the T_surface/T_water coupler naming debt
./docs/ocean/fidelity/autonomous_progress.md:389:  (CLAUDE.md open debt). CONFIRMED failing with my eos/model changes stashed (git stash + re-run).
./docs/ocean/fidelity/autonomous_progress.md-390-  Out of Phase-G ocean-recipe scope; flagged for the dedicated naming-cleanup PR.
./docs/ocean/fidelity/autonomous_progress.md-391-NEXT: Q8 — decompose latlon_cgrid_ocean_baroclinic_tendencies (the 1299-LOC fn in
./docs/ocean/fidelity/autonomous_progress.md-392-ocean/dynamics/ocean_pe_latlon_cgrid.py) into named substages; bit-identical-on-frozen-ACC-state
./docs/ocean/fidelity/autonomous_progress.md-393-gate; then drop its LOC_ALLOW_LIST entry in test_clarity_guards.py.
./docs/ocean/fidelity/autonomous_progress.md-394-
--
./docs/ocean/fidelity/autonomous_progress.md-415-caller tests green (gate + diagnostics-closure + differentiability + WENO + PGF), partial-cells
./docs/ocean/fidelity/autonomous_progress.md-416-(3b/4/5/7) + tendency-probe green.
./docs/ocean/fidelity/autonomous_progress.md-417-**Pre-existing conservation-drift failures (NOT regressions) — verified by checking out the
./docs/ocean/fidelity/autonomous_progress.md-418-pre-decomposition function:** test_variable_bathymetry smooth-bathy (heat drift 3.03e-7 vs 1e-8)
./docs/ocean/fidelity/autonomous_progress.md-419-and test_realistic_coastlines island (2.76e-6 vs 1e-7) fail IDENTICALLY on the ORIGINAL function
./docs/ocean/fidelity/autonomous_progress.md:420:(same drift to all digits) -> bit-identical drift CONFIRMS the extraction changed nothing; the
./docs/ocean/fidelity/autonomous_progress.md-421-tolerances are simply tighter than the model's intrinsic drift on those setups. Flagged as a
./docs/ocean/fidelity/autonomous_progress.md-422-separate conservation-tolerance issue, out of Q8 scope.
./docs/ocean/fidelity/autonomous_progress.md-423-
./docs/ocean/fidelity/autonomous_progress.md-424----
./docs/ocean/fidelity/autonomous_progress.md-425-
--
scripts/cluster/scaling_levante/README.md-72-| File | What |
scripts/cluster/scaling_levante/README.md-73-|---|---|
scripts/cluster/scaling_levante/README.md-74-| `gpu_scaling.sbatch` | ATM (cube AMIP physics, single-process multi-GPU) + OCEAN (`bench_ocean_latlon_spmd_scaling.py`, 1/2/4 A100 with parity+conservation smoke) on one GPU node. Plain GPU env — no mpi4jax. |
scripts/cluster/scaling_levante/README.md-75-| `gpu_multinode_scaling.sbatch` | MULTI-NODE route-B lanes over NCCL/IB (SLURM auto-detected `jax.distributed`): A = cube `--cs-spmd` (6 GPU / 2 nodes), C = atm lat-lon `--multicontroller` (8 GPU), D = ocean `--multicontroller` (8 GPU). |
scripts/cluster/scaling_levante/README.md-76-| `cpu_scaling.sbatch` | ATM + OCEAN CPU-MPI rank ladders on `compute` nodes (`legoesm-mpi` env), with fail-fast smokes. |
scripts/cluster/scaling_levante/README.md:77:| `diagnosis.sbatch` | BOTTLENECK diagnosis via `scripts/bench/run_scaling_diagnosis.py` — per-phase halo bandwidth, reduction latency, roofline, compute/comm overlap, and the **static collective census** (collective-permute + all-reduce + all-gather per step) the throughput jobs do NOT capture. Climbs the cube face-shard `1 2 3` ladder so the message-count-vs-shard curve — the LATENCY-bound anti-scaling signal — is recorded. `sbatch --export=ALL,MODE=census` for counts only. Derecho twin: `scaling_derecho/diagnosis.pbs`. |
scripts/cluster/scaling_levante/README.md-78-
scripts/cluster/scaling_levante/README.md-79-`_env.sh` now also carries the NCCL-over-IB defaults for the route-B lanes
scripts/cluster/scaling_levante/README.md-80-(`NCCL_IB_HCA=mlx5`, `NCCL_SOCKET_IFNAME=ib0`, `NCCL_NET_GDR_LEVEL=PHB`,
scripts/cluster/scaling_levante/README.md-81-`NCCL_CROSS_NIC=1`) — independent of, and coexisting with, the UCX/MPI
scripts/cluster/scaling_levante/README.md-82-settings used by route-A. First multi-node run: `NCCL_DEBUG=INFO` must show
--
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-17-# same steps/warmup as the s9 ladder.  Weak pairs at matched cells/GPU
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-18-# (81.9k / 41.0k / 20.5k) are computed ONLY from these rows vs 26600095.
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-19-#
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-20-# Falsifiability, written BEFORE submit:
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-21-#   numbers : s8-lloyd0 np8/16/32 steady_median_ms
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:22:#   CONFIRM (a matched-tile scale-out term exists): s9/s8 ratios at
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-23-#             matched cells/GPU stay well above 1 (prior draft saw
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-24-#             1.80/1.35/1.41 on the CONFOUNDED pairs)
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:25:#   REFUTE  : ratios collapse toward ~1.0 -> the draft's "term" was the
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-26-#             Lloyd-mesh confound, and MPAS-GPU weak scaling is near
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-27-#             ideal at matched tile.
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-28-# Mesh: prewarmed into LEGOESM_MESH_CACHE_DIR (subdiv-8 is below the
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-29-# big-mesh refuse threshold, so a cache miss falls back to in-process
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-30-# builds — slower, still correct).
--
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-223-
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-224-# Lane T (RUN_TUNE=1): comm-tuning A/B ladder — Derecho twin (see
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-225-# gpu_multinode_scaling.pbs lane T + docs/performance/scaling/
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-226-# spmd_message_census_2026-07-08.md). Arms: base (control, same allocation),
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-227-# fused (LEGOESM_LATLON_SPMD_FUSED_HALO=1 — bit-identical multi-pad packing,
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:228:# trace receipt atm 41->29 CPs/step), xla (collective-permute combining +
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-229-# pipelined p2p), pgle (profile-guided latency estimation; recompiles after
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-230-# the profiling runs — bench-jit-safe, not for AOT jobs). Outputs land under
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-231-# _ab_tuning/ which aggregate_bcw_scaling SKIPS (A/B receipts never join the
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-232-# scaling curves); compare arms via steady_median_ms / sypd per JSONL row.
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-233-RUN_TUNE="${RUN_TUNE:-0}"
--
scripts/cluster/scaling_levante/diagnosis.sbatch-12-# BOTTLENECK DIAGNOSIS on a Levante GPU node (4x A100-80GB, NVLink).
scripts/cluster/scaling_levante/diagnosis.sbatch-13-# Levante twin of scaling_derecho/diagnosis.pbs — identical tool + ladder.
scripts/cluster/scaling_levante/diagnosis.sbatch-14-#
scripts/cluster/scaling_levante/diagnosis.sbatch-15-# Runs scripts/bench/run_scaling_diagnosis.py — the per-phase bottleneck tool
scripts/cluster/scaling_levante/diagnosis.sbatch-16-# (halo bandwidth, reduction latency, roofline, compute/comm overlap, and the
scripts/cluster/scaling_levante/diagnosis.sbatch:17:# STATIC collective census: collective-permute + all-reduce + all-gather per
scripts/cluster/scaling_levante/diagnosis.sbatch-18-# step) — which the throughput jobs (gpu_scaling.sbatch) do NOT capture.
scripts/cluster/scaling_levante/diagnosis.sbatch-19-#
scripts/cluster/scaling_levante/diagnosis.sbatch-20-# Why this matters: the measured f64==f32 GPU strong-scaling curves mean the
scripts/cluster/scaling_levante/diagnosis.sbatch-21-# multi-GPU leg is LATENCY-bound, so the per-step MESSAGE COUNT — captured by
scripts/cluster/scaling_levante/diagnosis.sbatch-22-# the census below — is the lever, not the byte volume.  Cube face-sharding
--
./tests/parallel/test_latlon_spmd_fused_halo.py-6-1. BIT-identity vs the per-field wall pads (mixed trailing shapes, mixed
./tests/parallel/test_latlon_spmd_fused_halo.py-7-   boundary constants, mixed dtypes) — the fusion is packing, never
./tests/parallel/test_latlon_spmd_fused_halo.py-8-   arithmetic.
./tests/parallel/test_latlon_spmd_fused_halo.py-9-2. End-to-end bit-identity of the FULL sharded ocean step with the flag
./tests/parallel/test_latlon_spmd_fused_halo.py-10-   on vs off.
./tests/parallel/test_latlon_spmd_fused_halo.py:11:3. The point, mechanically: compiled ``collective-permute`` count drops
./tests/parallel/test_latlon_spmd_fused_halo.py-12-   (one pair per direction per dtype group instead of one per field).
./tests/parallel/test_latlon_spmd_fused_halo.py-13-
./tests/parallel/test_latlon_spmd_fused_halo.py-14-Also gates the two M2-leftover message-aggregation levers (M4 quick wins):
./tests/parallel/test_latlon_spmd_fused_halo.py-15-
./tests/parallel/test_latlon_spmd_fused_halo.py-16-4. ``reconstruct_vface_lower_multi`` — the fused v-carrier boundary-row
./tests/parallel/test_latlon_spmd_fused_halo.py-17-   reconstruction (v + v_mask in ONE ppermute per dtype group inside the
./tests/parallel/test_latlon_spmd_fused_halo.py-18-   sharded ocean step): bit-identity vs the per-field
./tests/parallel/test_latlon_spmd_fused_halo.py:19:   ``reconstruct_vface_lower`` + the compiled collective-permute drop.
./tests/parallel/test_latlon_spmd_fused_halo.py-20-5. ``eta_floor._global_sum_pair`` — the batched SPMD psum pair (ONE packed
./tests/parallel/test_latlon_spmd_fused_halo.py-21-   ``psum`` via ``batch_psum_spmd`` instead of two): bit-identity vs the
./tests/parallel/test_latlon_spmd_fused_halo.py-22-   separate psums + exactly one compiled all-reduce.
./tests/parallel/test_latlon_spmd_fused_halo.py-23-
./tests/parallel/test_latlon_spmd_fused_halo.py-24-Run: ``XLA_FLAGS=--xla_force_host_platform_device_count=4 \
--
./tests/parallel/test_latlon_spmd_fused_halo.py-130-            jnp.zeros((N_LAT, N_LON)))
./tests/parallel/test_latlon_spmd_fused_halo.py-131-
./tests/parallel/test_latlon_spmd_fused_halo.py-132-
./tests/parallel/test_latlon_spmd_fused_halo.py-133-def _count_ppermutes(hlo_text: str) -> int:
./tests/parallel/test_latlon_spmd_fused_halo.py-134-    return sum(1 for line in hlo_text.splitlines()
./tests/parallel/test_latlon_spmd_fused_halo.py:135:               if "collective-permute" in line and "done" not in line)
./tests/parallel/test_latlon_spmd_fused_halo.py-136-
./tests/parallel/test_latlon_spmd_fused_halo.py-137-
./tests/parallel/test_latlon_spmd_fused_halo.py-138-def test_fused_cuts_collective_count():
./tests/parallel/test_latlon_spmd_fused_halo.py-139-    """5 same-dtype fields: per-field = 10 ppermutes (2/field), fused = 2."""
./tests/parallel/test_latlon_spmd_fused_halo.py-140-    mesh = _mesh()
--
./tests/parallel/test_latlon_spmd_fused_halo.py-226-        assert not np.asarray(f_out)[-1].any()
./tests/parallel/test_latlon_spmd_fused_halo.py-227-
./tests/parallel/test_latlon_spmd_fused_halo.py-228-
./tests/parallel/test_latlon_spmd_fused_halo.py-229-def test_vface_multi_reconstruct_cuts_collective_count():
./tests/parallel/test_latlon_spmd_fused_halo.py-230-    """2 same-dtype carriers (the production v + v_mask pair): per-field = 2
./tests/parallel/test_latlon_spmd_fused_halo.py:231:    collective-permutes, fused = 1."""
./tests/parallel/test_latlon_spmd_fused_halo.py-232-    from legoesm.parallel.latlon_spmd import (
./tests/parallel/test_latlon_spmd_fused_halo.py-233-        latlon_band_perms,
./tests/parallel/test_latlon_spmd_fused_halo.py-234-        reconstruct_vface_lower,
./tests/parallel/test_latlon_spmd_fused_halo.py-235-        reconstruct_vface_lower_multi,
./tests/parallel/test_latlon_spmd_fused_halo.py-236-    )
--
./tests/parallel/test_latlon_spmd_fused_halo.py-311-
./tests/parallel/test_latlon_spmd_fused_halo.py-312-@pytest.mark.parametrize("n_steps", [2])
./tests/parallel/test_latlon_spmd_fused_halo.py-313-def test_ocean_step_bit_identical_with_fusion(monkeypatch, n_steps):
./tests/parallel/test_latlon_spmd_fused_halo.py-314-    """The FULL sharded ocean step: fusion on vs off must be BIT-identical
./tests/parallel/test_latlon_spmd_fused_halo.py-315-    (packing only, no arithmetic) — and the compiled step must carry fewer
./tests/parallel/test_latlon_spmd_fused_halo.py:316:    collective-permutes with fusion on (the aggregation actually fires on
./tests/parallel/test_latlon_spmd_fused_halo.py-317-    the production pad_multi sites)."""
./tests/parallel/test_latlon_spmd_fused_halo.py-318-    from legoesm.grids.latlon import create_latlon_grid
./tests/parallel/test_latlon_spmd_fused_halo.py-319-    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
./tests/parallel/test_latlon_spmd_fused_halo.py-320-        LatLonCGridOceanModel,
./tests/parallel/test_latlon_spmd_fused_halo.py-321-    )
--
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-24-#   solo_post : solo again AFTER phase B (brackets ordering/thermal
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-25-#               drift; contrast uses mean of the two solos)
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-26-#
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-27-# Falsifiability, written BEFORE submit:
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-28-#   numbers : 2 solo + 4 replica steady_median_ms
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:29:#   CONFIRM : max(replica) <= 1.10 x mean(solo) -> guaranteed aggregate
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-30-#             >= 4/1.10 = 3.64x the 32-GPU solo rate (~19.9 GC/s if solo
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-31-#             reproduces 5.47) = ~3.3x the observed 128-GPU
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-32-#             single-trajectory rate.  NOT "4x": 1.10 is the bar, the
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-33-#             margin below it is the measured contention.
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:34:#   REFUTE  : any replica > 1.10x solo -> contention term, quantified
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-35-#             per replica.
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-36-# Protocol: config identical to job 26600095 np32 rung (sfc, lloyd 0,
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-37-# f32, padded-128 reorder) EXCEPT steps 5000 / warmup 100 so the stepping
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-38-# window (~60 s at 12.5 ms/step) dwarfs launch skew between replicas --
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-39-# overlap is EVIDENCED, not assumed, by the per-step Start/End + NodeList
--
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-10-#SBATCH --time=01:30:00
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-11-#SBATCH --output=ll128_comb.%j.log
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-12-# CP-COMBINING A/B at LL2048@128 (the measured/bound ~2.9 gap).
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-13-# The calibrated bound (census 41 CP + 1 AR/step, exact bytes 18.5 MB/dev,
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-14-# IB 26.3us/23.5GB/s, nd=1 same-tile compute 1.659 ms) models 1.85-1.89 ms;
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:15:# measured is 5.58 (job 26534060). Leading PLAUSIBLE mechanism: effective
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-16-# per-CP overhead (launch+schedule+sync) >> raw fabric latency across 41
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-17-# dependency-chained exchanges — the arXiv:2607.16100 regime. Lever:
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-18-# COMBINE independent CPs into fewer, larger messages.
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-19-# NOTE: the closed-levers null for these flags was the OCEAN lane
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-20-# (reduction-dominated); this is the first atm-latlon test — not a rerun
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-21-# of a closed null.
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-22-# Falsifiability, BEFORE submit — arms byte-matched to 26534060 protocol:
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-23-#   A control (default flags)      : expect ~5.6 ms
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:24:#   B +cp-combine 8MB threshold    : CONFIRM lever if >=10% under A
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-25-#   C +combine +pipelined-p2p      : scheduling interaction
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:26:#   REFUTE if B,C within 2% of A -> overhead is not combinable-CP count;
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-27-#   next hypothesis = unoverlapped serial chain (scheduling lever).
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-28-set -uo pipefail
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-29-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-30-export JAX_PLATFORMS=cuda,cpu
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-31-export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
--
./packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py-627-        dz_surface = None
./packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py-628-        if (getattr(tke_cfg, "veros_dz_slots", False)
./packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py-629-                or getattr(tke_cfg, "tke_surface_bc_level",
./packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py-630-                           "interior_pinned") == "nemo_z0"):
./packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py-631-            dz_surface = (-z_coord.z_full_ref[0]) * J
./packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py:632:        # Veros tke_mxl_choice=1 distance-to-boundary cap (tke.py:43-47):
./packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py-633-        # the buoyancy mixing length may not exceed the distance to the
./packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py-634-        # surface / seafloor. Computed once from the STATIC reference geometry
./packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py-635-        # (interior interface heights + centre spacing) and the per-column
./packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py-636-        # ocean depth (H_bathy = Veros ``ht``). Only choice=1 needs it;
./packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py-637-        # choice=2 is bounded by the MITgcm/OPA recursion in
--
./tests/parallel/test_ppermute_multiface.py-326-    monkeypatch.delenv("LEGOESM_SPMD_FORCE_ALLGATHER")
./tests/parallel/test_ppermute_multiface.py-327-    assert cx.select_exchange_backend(8, 1, 2) is True
./tests/parallel/test_ppermute_multiface.py-328-
./tests/parallel/test_ppermute_multiface.py-329-
./tests/parallel/test_ppermute_multiface.py-330-def test_routing_emits_collective_permute_not_allgather():
./tests/parallel/test_ppermute_multiface.py:331:    """The compiled multiface h1+h2 programs contain collective-permute
./tests/parallel/test_ppermute_multiface.py-332-    and no all-gather; the one-face kernel still owns halo=1 at 6
./tests/parallel/test_ppermute_multiface.py-333-    devices (cache key distinguishes the variants)."""
./tests/parallel/test_ppermute_multiface.py-334-    n = 8
./tests/parallel/test_ppermute_multiface.py-335-    data = _ramp_3d(n)
./tests/parallel/test_ppermute_multiface.py-336-    mesh2 = _mesh(2)
./tests/parallel/test_ppermute_multiface.py-337-    cx.set_ppermute_default(True)
./tests/parallel/test_ppermute_multiface.py-338-    for halo in (1, 2):
./tests/parallel/test_ppermute_multiface.py-339-        txt = jax.jit(
./tests/parallel/test_ppermute_multiface.py-340-            lambda d, h=halo: cx.explicit_pad_halo(d, mesh2, halo=h)
./tests/parallel/test_ppermute_multiface.py-341-        ).lower(data).compile().as_text()
./tests/parallel/test_ppermute_multiface.py:342:        assert "collective-permute" in txt, f"halo={halo}: ppermute missing"
./tests/parallel/test_ppermute_multiface.py-343-        assert "all-gather" not in txt, f"halo={halo}: all-gather present"
./tests/parallel/test_ppermute_multiface.py-344-    # Variant resolution: one-face kernel reserved for h1@6dev only.
./tests/parallel/test_ppermute_multiface.py-345-    assert cx._select_variant(True, 1, 6) == "ppermute_oneface"
./tests/parallel/test_ppermute_multiface.py-346-    assert cx._select_variant(True, 2, 6) == "ppermute_multiface"
./tests/parallel/test_ppermute_multiface.py-347-    assert cx._select_variant(True, 1, 2) == "ppermute_multiface"
--
./tests/parallel/test_ppermute_multiface.py-439-def test_scan_hot_path_has_zero_fullcube_allgathers(n_devices):
./tests/parallel/test_ppermute_multiface.py-440-    """THE mechanical tripwire for the replication bug class: the
./tests/parallel/test_ppermute_multiface.py-441-    compiled scan hot path under the ppermute backend must contain ZERO
./tests/parallel/test_ppermute_multiface.py-442-    all-gather ops with full-cube face extent (and, on this pure
./tests/parallel/test_ppermute_multiface.py-443-    halo+stencil step, zero all-gathers at all), while actually routing
./tests/parallel/test_ppermute_multiface.py:444:    collective-permute."""
./tests/parallel/test_ppermute_multiface.py-445-    n, c = 8, 2
./tests/parallel/test_ppermute_multiface.py-446-    mesh = _mesh(n_devices)
./tests/parallel/test_ppermute_multiface.py-447-    cx.activate_spmd_halo_backend(mesh, n=n, nlev=c)
./tests/parallel/test_ppermute_multiface.py-448-    try:
./tests/parallel/test_ppermute_multiface.py-449-        txt = _scan_runner_hlo(mesh, n, c)
./tests/parallel/test_ppermute_multiface.py-450-    finally:
./tests/parallel/test_ppermute_multiface.py-451-        cx.deactivate_spmd_halo_backend()
./tests/parallel/test_ppermute_multiface.py-452-    assert cx.find_fullcube_allgathers(txt, n_faces=6, n=n) == []
./tests/parallel/test_ppermute_multiface.py-453-    cx.assert_no_fullcube_allgather(txt, n=n, context="scan hot path")
./tests/parallel/test_ppermute_multiface.py:454:    assert "collective-permute" in txt
./tests/parallel/test_ppermute_multiface.py-455-    assert "all-gather" not in txt  # codex acceptance: hot-path count 0
./tests/parallel/test_ppermute_multiface.py-456-
./tests/parallel/test_ppermute_multiface.py-457-
./tests/parallel/test_ppermute_multiface.py-458-def test_hlo_guard_flags_real_allgather_program():
./tests/parallel/test_ppermute_multiface.py-459-    """Non-vacuous tripwire, REAL-program violation: the explicit
--
./tests/parallel/test_ppermute_multiface.py-491-    assert flag("%agd = f32[6,4,24,8]{3,2,1,0} all-gather-done(%ags)")
./tests/parallel/test_ppermute_multiface.py-492-    assert flag("%agd.1 = f32[6,26,26,8]{3,2,1,0} all-gather-done("
./tests/parallel/test_ppermute_multiface.py-493-                "(f32[3,26,26,8], f32[6,26,26,8]) %ags.1)")
./tests/parallel/test_ppermute_multiface.py-494-    # Volume false-pass surface: face dim folded away but >= 6*n*n elems.
./tests/parallel/test_ppermute_multiface.py-495-    assert flag("%ag.2 = f32[144,64]{1,0} all-gather(f32[72,64] %q)", n=24)
./tests/parallel/test_ppermute_multiface.py:496:    # Benign: collective-permute is the expected op.
./tests/parallel/test_ppermute_multiface.py:497:    assert not flag("%cp = f32[24,8]{1,0} collective-permute(f32[24,8] %s)")
./tests/parallel/test_ppermute_multiface.py-498-    # Benign: small partial gather without face extent or cube volume.
./tests/parallel/test_ppermute_multiface.py-499-    assert not flag("%ag.3 = f32[2,24,8]{2,1,0} all-gather(f32[1,24,8] %r)",
./tests/parallel/test_ppermute_multiface.py-500-                    n=24)
./tests/parallel/test_ppermute_multiface.py-501-    # Benign: 6-element scalar-stats gather (one element per face).
./tests/parallel/test_ppermute_multiface.py-502-    assert not flag("%ag.4 = f32[6]{0} all-gather(f32[3] %t)")
--
./tests/parallel/test_ppermute_multiface.py-541-        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
./tests/parallel/test_ppermute_multiface.py-542-    txt = jax.jit(
./tests/parallel/test_ppermute_multiface.py-543-        lambda d: cx.explicit_pad_halo(d, mesh1, halo=2)
./tests/parallel/test_ppermute_multiface.py-544-    ).lower(data).compile().as_text()
./tests/parallel/test_ppermute_multiface.py-545-    assert "all-gather" not in txt
./tests/parallel/test_ppermute_multiface.py:546:    assert "collective-permute" not in txt
./tests/parallel/test_ppermute_multiface.py-547-
./tests/parallel/test_ppermute_multiface.py-548-
./tests/parallel/test_ppermute_multiface.py-549-def teardown_module(_):
./tests/parallel/test_ppermute_multiface.py-550-    cx.set_ppermute_default(False)
./tests/parallel/test_ppermute_multiface.py-551-    cx._cache.clear()
--
./packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py-789-            # mode-1 (1-fi) law (codex HIGH).
./packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py-790-            ice_frac = (ice_frac if _tke_eice == 1
./packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py-791-                        else jnp.minimum(4.0 * ice_frac, 1.0))
./packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py-792-        lat_deg = jnp.degrees(mesh.latCell)
./packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py-793-
./packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py:794:        # Veros tke_mxl_choice=1 distance-to-boundary cap (mirrors the lat-lon
./packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py-795-        # k_profiles branch): the buoyancy mixing length may not exceed the
./packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py-796-        # distance to surface/seafloor.  Built from the STATIC reference geometry
./packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py-797-        # + per-column ocean depth.  choice=2 (default) is bounded by the
./packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py-798-        # MITgcm/OPA recursion and needs no cap.
./packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py-799-        boundary_cap = None
--
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-553-def veros_mxl_choice1_boundary_cap(
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-554-    z_interface: jnp.ndarray,
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-555-    dz_half: jnp.ndarray,
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-556-    column_depth: jnp.ndarray,
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-557-) -> jnp.ndarray:
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:558:    r"""Veros ``tke_mxl_choice=1`` distance-to-boundary cap (tke.py:39-47).
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-559-
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-560-    Veros bounds the buoyancy mixing length by the geometric distance to the
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-561-    nearest boundary so a parcel can never mix across more than its distance
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-562-    to the surface or the seafloor:
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-563-
--
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-1728-    if positivity == "veros_surface_correction" and cfg.n2_mode != "adiabatic":
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-1729-        # The Veros surface correction lets the interior TKE carry a negative
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-1730-        # energy debt; only the signed-N² adiabatic buoyancy length handles
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-1731-        # it (sqrt(max(0,e)) clamp). tke_mxl_choice ∈ {1, 2, 3} are all
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-1732-        # debt-safe on the adiabatic path: choice=2 via _veros_buoyancy_length's
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:1733:        # recursion, choice=1 via the distance-to-boundary cap
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-1734-        # (veros_mxl_choice1_boundary_cap, supplied by the orchestrator),
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-1735-        # choice=3 via its own double-where sqrt(2e) + the lup/ldown caps. The
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-1736-        # in-situ N² branch (n2_mode != 'adiabatic') is NOT debt-safe (raw e
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-1737-        # inside the closed-form sqrt) — fail loudly at config time.
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-1738-        raise ValueError(
--
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-2021-                f"{getattr(cfg, 'positivity', 'floor')!r}."
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-2022-            )
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-2023-        # tke_mxl_choice ∈ {1, 2} are both debt-safe under the post-mixing
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-2024-        # surface correction. choice=2 is bounded by the MITgcm/OPA recursion
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-2025-        # (_veros_buoyancy_length); choice=1 is bounded by the Veros
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:2026:        # distance-to-boundary cap (veros_mxl_choice1_boundary_cap, wired in by
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-2027-        # the orchestrator). Both match Veros (only global_1deg selects
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-2028-        # choice=1). compute_mixing_lengths raises on any other value.
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-2029-        #
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-2030-        # Mixed-oracle guard (RELAXED for lc 2026-07-16: the Langmuir source is
./packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py-2031-        # now computed in tke_set_diffusivities and applied pre-solve inside
--
./packages/ocean/legoesm/ocean/fidelity/veros_global_1deg_recipe.py-60-     4deg/flexible block (``TKEConfig.tke_mxl_choice`` dispatches it in
./packages/ocean/legoesm/ocean/fidelity/veros_global_1deg_recipe.py-61-     ``vertical_mixing/tke.py``; everything else — c_k=0.1, c_eps=0.7,
./packages/ocean/legoesm/ocean/fidelity/veros_global_1deg_recipe.py-62-     alpha_tke=30, mxl_min=1e-8, kappaM_min=2e-4, kappaH_min=2e-5, kappaH
./packages/ocean/legoesm/ocean/fidelity/veros_global_1deg_recipe.py-63-     profile, superbee advection — is the verbatim 4deg block).
./packages/ocean/legoesm/ocean/fidelity/veros_global_1deg_recipe.py-64-
./packages/ocean/legoesm/ocean/fidelity/veros_global_1deg_recipe.py:65:     ``tke_mxl_choice=1`` is the DEBT-SAFE Veros distance-to-boundary
./packages/ocean/legoesm/ocean/fidelity/veros_global_1deg_recipe.py-66-     limiter (``veros/core/tke.py:43-47``): the buoyancy mixing length is
./packages/ocean/legoesm/ocean/fidelity/veros_global_1deg_recipe.py-67-     capped by ``min(-zw + dzw/2, ht + zw)`` — the distance to the surface
./packages/ocean/legoesm/ocean/fidelity/veros_global_1deg_recipe.py-68-     and to the seafloor — before the ``mxl_min`` floor. legoESM imports
./packages/ocean/legoesm/ocean/fidelity/veros_global_1deg_recipe.py-69-     this faithfully via :func:`veros_mxl_choice1_boundary_cap`, wired in by
./packages/ocean/legoesm/ocean/fidelity/veros_global_1deg_recipe.py-70-     the K-profile orchestrator from the static interface geometry + the
--
./tests/unit/test_cdgrid_fv3_regression.py-4384-        pins:
./tests/unit/test_cdgrid_fv3_regression.py-4385-
./tests/unit/test_cdgrid_fv3_regression.py-4386-        - The 4 expected n-values in {16, 24, 36, 48}.
./tests/unit/test_cdgrid_fv3_regression.py-4387-        - All peak-error cells on face 3 (matches iter-780
./tests/unit/test_cdgrid_fv3_regression.py-4388-          measurement).
./tests/unit/test_cdgrid_fv3_regression.py:4389:        - The GC-distance-to-cube-vertex values within a generous
./tests/unit/test_cdgrid_fv3_regression.py-4390-          envelope 5° to 40°.
./tests/unit/test_cdgrid_fv3_regression.py-4391-        - The summary block lines present.
./tests/unit/test_cdgrid_fv3_regression.py-4392-
./tests/unit/test_cdgrid_fv3_regression.py-4393-        Runtime: ~0.3s file parse only.
./tests/unit/test_cdgrid_fv3_regression.py-4394-        """
--
./tests/parallel/test_cube_tile_native_segment.py-13-  (cross-lane; ``kt=1`` tiled meshes are refused BY DESIGN —
./tests/parallel/test_cube_tile_native_segment.py-14-  ``cubesphere_exchange._build_tiled_tables``: kt=1 IS the face exchange —
./tests/parallel/test_cube_tile_native_segment.py-15-  so at 6 devices the production path is the face-SPMD lane and the
./tests/parallel/test_cube_tile_native_segment.py-16-  comparison runs across lanes),
./tests/parallel/test_cube_tile_native_segment.py-17-* compiled-HLO receipts: ZERO ``all-gather`` anywhere in the segment
./tests/parallel/test_cube_tile_native_segment.py:18:  executable; ``collective-permute`` (tile halos) present; collective
./tests/parallel/test_cube_tile_native_segment.py-19-  counts of the 5-step segment EQUAL the 4-step segment's (the loop body
./tests/parallel/test_cube_tile_native_segment.py-20-  is SHARED; the fixed extra copies are the ONE dtype-unrolled leading
./tests/parallel/test_cube_tile_native_segment.py-21-  step + XLA's boundary peel — 2 total copies @n=2 vs 3 @n=5, job
./tests/parallel/test_cube_tile_native_segment.py-22-  8970742 — so 4-vs-5 compares past them, where a real per-step
./tests/parallel/test_cube_tile_native_segment.py-23-  unroll/regather would still scale counts),
--
./tests/parallel/test_cube_tile_native_segment.py-314-def test_segment_hlo_no_allgather_collectives_once():
./tests/parallel/test_cube_tile_native_segment.py-315-    """HLO receipts for the WHOLE segment executable:
./tests/parallel/test_cube_tile_native_segment.py-316-
./tests/parallel/test_cube_tile_native_segment.py-317-    * ZERO ``all-gather`` (no full-face re-gather anywhere — the M3b
./tests/parallel/test_cube_tile_native_segment.py-318-      property, asserted on the compiled program, not inferred),
./tests/parallel/test_cube_tile_native_segment.py:319:    * ``collective-permute`` present (the in-stage tile halos really run),
./tests/parallel/test_cube_tile_native_segment.py-320-    * ``all-reduce`` present (the in-stage telescoping mass fixer's psum —
./tests/parallel/test_cube_tile_native_segment.py-321-      the production config's ONE algorithmic global reduction),
./tests/parallel/test_cube_tile_native_segment.py-322-    * collective COUNTS of the 5-step segment == the 4-step segment's:
./tests/parallel/test_cube_tile_native_segment.py-323-      the scan body is SHARED between iterations, so the counts are
./tests/parallel/test_cube_tile_native_segment.py-324-      n-independent past the FIXED extra body copies — the ONE
--
./tests/parallel/test_cube_tile_native_segment.py-341-    hlo5 = _dry_compiled(N_STEPS).as_text()
./tests/parallel/test_cube_tile_native_segment.py-342-
./tests/parallel/test_cube_tile_native_segment.py-343-    for tag, hlo in (("4-step", hlo4), ("5-step", hlo5)):
./tests/parallel/test_cube_tile_native_segment.py-344-        assert hlo.count("all-gather") == 0, (
./tests/parallel/test_cube_tile_native_segment.py-345-            f"full-face all-gather in the {tag} segment")
./tests/parallel/test_cube_tile_native_segment.py:346:    assert "collective-permute" in hlo5, "tile halos missing from segment"
./tests/parallel/test_cube_tile_native_segment.py-347-    assert "all-reduce" in hlo5, "in-stage mass-fixer psum missing"
./tests/parallel/test_cube_tile_native_segment.py-348-    assert " while(" in hlo5, "5-step segment did not compile as a scan loop"
./tests/parallel/test_cube_tile_native_segment.py:349:    for op in ("collective-permute", "all-reduce", "all-gather",
./tests/parallel/test_cube_tile_native_segment.py-350-               "all-to-all"):
./tests/parallel/test_cube_tile_native_segment.py-351-        c4, c5 = hlo4.count(op), hlo5.count(op)
./tests/parallel/test_cube_tile_native_segment.py-352-        assert c5 == c4, (
./tests/parallel/test_cube_tile_native_segment.py-353-            f"{op}: {c5} in 5-step segment vs {c4} in 4-step segment — "
./tests/parallel/test_cube_tile_native_segment.py-354-            f"the scan body must be shared (counts n-independent past "
--
./tests/parallel/test_segment_sharding_device_config.py-443-            _count_collective_permute_ops,
./tests/parallel/test_segment_sharding_device_config.py-444-        )
./tests/parallel/test_segment_sharding_device_config.py-445-        hlo = "\n".join([
./tests/parallel/test_segment_sharding_device_config.py-446-            # sync form: opcode application counted once, result-name
./tests/parallel/test_segment_sharding_device_config.py-447-            # mentions ignored
./tests/parallel/test_segment_sharding_device_config.py:448:            "%collective-permute.5 = f32[3,24,24]{2,1,0}"
./tests/parallel/test_segment_sharding_device_config.py:449:            " collective-permute(f32[3,24,24]{2,1,0} %p),"
./tests/parallel/test_segment_sharding_device_config.py-450-            " source_target_pairs={{0,1},{1,0}}",
./tests/parallel/test_segment_sharding_device_config.py:451:            "%add.1 = f32[] add(f32[] %collective-permute.5, f32[] %c)",
./tests/parallel/test_segment_sharding_device_config.py-452-            # async pair: -start/-done are NOT double-counted as the
./tests/parallel/test_segment_sharding_device_config.py-453-            # plain opcode
./tests/parallel/test_segment_sharding_device_config.py:454:            "%cps = (f32[3],f32[3]) collective-permute-start(f32[3] %q)",
./tests/parallel/test_segment_sharding_device_config.py:455:            "%cpd = f32[3] collective-permute-done((f32[3],f32[3]) %cps)",
./tests/parallel/test_segment_sharding_device_config.py:456:            "%cps2 = (f32[3],f32[3]) collective-permute-start(f32[3] %r)",
./tests/parallel/test_segment_sharding_device_config.py:457:            "%cpd2 = f32[3] collective-permute-done((f32[3],f32[3]) %cps2)",
./tests/parallel/test_segment_sharding_device_config.py-458-        ])
./tests/parallel/test_segment_sharding_device_config.py-459-        counts = _count_collective_permute_ops(hlo)
./tests/parallel/test_segment_sharding_device_config.py:460:        assert counts["collective-permute"] == 1
./tests/parallel/test_segment_sharding_device_config.py:461:        assert counts["collective-permute-start"] == 2
./tests/parallel/test_segment_sharding_device_config.py:462:        assert counts["collective-permute-done"] == 2
./tests/parallel/test_segment_sharding_device_config.py-463-
./tests/parallel/test_segment_sharding_device_config.py-464-    def test_zero_on_collective_free_hlo(self):
./tests/parallel/test_segment_sharding_device_config.py-465-        from scripts.bench.run_levante_gpu_scaling import (
./tests/parallel/test_segment_sharding_device_config.py-466-            _count_collective_permute_ops,
./tests/parallel/test_segment_sharding_device_config.py-467-        )
./tests/parallel/test_segment_sharding_device_config.py-468-        counts = _count_collective_permute_ops(
./tests/parallel/test_segment_sharding_device_config.py-469-            "%add.0 = f32[8] add(f32[8] %a, f32[8] %b)\n"
./tests/parallel/test_segment_sharding_device_config.py-470-            "%ag = f32[6,8] all-gather(f32[3,8] %c), dimensions={0}\n"
./tests/parallel/test_segment_sharding_device_config.py-471-        )
./tests/parallel/test_segment_sharding_device_config.py-472-        assert counts == {
./tests/parallel/test_segment_sharding_device_config.py:473:            "collective-permute": 0,
./tests/parallel/test_segment_sharding_device_config.py:474:            "collective-permute-start": 0,
./tests/parallel/test_segment_sharding_device_config.py:475:            "collective-permute-done": 0,
./tests/parallel/test_segment_sharding_device_config.py-476-        }
./tests/parallel/test_segment_sharding_device_config.py-477-
./tests/parallel/test_segment_sharding_device_config.py-478-
./tests/parallel/test_segment_sharding_device_config.py-479-class TestSegmentDeviceConfigNonePath:
./tests/parallel/test_segment_sharding_device_config.py-480-    """device_config=None: byte-identical legacy behaviour (any device count)."""
--
./docs/ocean/experiments/dino_acc_solver_diagnostic_and_controls.md-328-DINO ACC over-deepening is an **eddy-saturation closure** problem: the Visbeck κ_GM
./docs/ocean/experiments/dino_acc_solver_diagnostic_and_controls.md-329-(α=0.015, 200–2000 m²/s) under-predicts the eddy diffusivity the 1° channel needs
./docs/ocean/experiments/dino_acc_solver_diagnostic_and_controls.md-330-(~5000 m²/s). FIX candidate = raise `visbeck_kappa_min` to ~5000.
./docs/ocean/experiments/dino_acc_solver_diagnostic_and_controls.md-331-
./docs/ocean/experiments/dino_acc_solver_diagnostic_and_controls.md-332-**…but the yr-4 confirmation FALSIFIES even this (the "smoke past the turnover"
./docs/ocean/experiments/dino_acc_solver_diagnostic_and_controls.md:333:lesson, again).** `GM_CONFIRM=1` (kpp+wright, 4 yr):
./docs/ocean/experiments/dino_acc_solver_diagnostic_and_controls.md-334-
./docs/ocean/experiments/dino_acc_solver_diagnostic_and_controls.md-335-| config | yr1 | yr2 | yr3 | yr4 |
./docs/ocean/experiments/dino_acc_solver_diagnostic_and_controls.md-336-| --- | --- | --- | --- | --- |
./docs/ocean/experiments/dino_acc_solver_diagnostic_and_controls.md-337-| Visbeck base | 45 | 84 | 232 | 604 |
./docs/ocean/experiments/dino_acc_solver_diagnostic_and_controls.md-338-| κ_min 5000 (floor or const) | 37 | **66** | **238** | **601** |
--
./tests/unit/test_bechtold.py-1022-    )
./tests/unit/test_bechtold.py-1023-
./tests/unit/test_bechtold.py-1024-
./tests/unit/test_bechtold.py-1025-# ---------------------------------------------------------------------------
./tests/unit/test_bechtold.py-1026-# IFS-faithfulness fixes (audit vs ecmwf-ifs/openifs): F1/F4/F5 shipped;
./tests/unit/test_bechtold.py:1027:# F6 REFUTED by SCM-RCE (mismapped coefficient) and reverted to the tuned value
./tests/unit/test_bechtold.py-1028-# ---------------------------------------------------------------------------
./tests/unit/test_bechtold.py-1029-
./tests/unit/test_bechtold.py-1030-from legoesm.atmosphere.physics.convection.bechtold import (  # noqa: E402
./tests/unit/test_bechtold.py-1031-    _ifs_updraft_mean_velocity,
./tests/unit/test_bechtold.py-1032-    _ifs_deep_turnover_scale,
--
./tests/ocean/unit/test_gm_taper_single_f_faithful.py:1:"""Certifies the GM/Redi DM95 taper is oracle-faithful (a REFUTED review finding).
./tests/ocean/unit/test_gm_taper_single_f_faithful.py-2-
./tests/ocean/unit/test_gm_taper_single_f_faithful.py-3-Oracle-review claim: legoESM applies the DM95 taper to the SLOPE (giving f² on
./tests/ocean/unit/test_gm_taper_single_f_faithful.py-4-the K33 diagonal) where Veros applies a single f to the whole tensor. The
./tests/ocean/unit/test_gm_taper_single_f_faithful.py:5:multi-angle investigation REFUTED this as a bug:
./tests/ocean/unit/test_gm_taper_single_f_faithful.py-6-
./tests/ocean/unit/test_gm_taper_single_f_faithful.py-7-  * The DEFAULT / oracle path is ``slope_scheme="triads"`` (raw slopes, a single
./tests/ocean/unit/test_gm_taper_single_f_faithful.py-8-    taper factor on K33 — matching Veros/pyOM2 ``isoneutral.py`` exactly). Every
./tests/ocean/unit/test_gm_taper_single_f_faithful.py-9-    Veros/NEMO fidelity recipe selects it.
./tests/ocean/unit/test_gm_taper_single_f_faithful.py-10-  * The f² appears ONLY on the non-default centered/cube path, where K11 is left
--
./docs/ocean/experiments/density_jacobian_pgf_mpas.md-1106-| Option | Old estimate | Updated assessment |
./docs/ocean/experiments/density_jacobian_pgf_mpas.md-1107-|--------|--------------|--------------------|
./docs/ocean/experiments/density_jacobian_pgf_mpas.md-1108-| (A) Accept ico-4 ceiling | 0 effort, no fix | **Still valid** as ship-as-is |
./docs/ocean/experiments/density_jacobian_pgf_mpas.md-1109-| **(B) Static ρ_ref(z)** | ~60 % to 1-yr | **Promoted to next experiment.** Now known to be the only option that attacks the seed *globally* rather than relocating the symptom.  Offline probe: 24× residual reduction. |
./docs/ocean/experiments/density_jacobian_pgf_mpas.md-1110-| (C) AHH08 PGF | ~80 % to 5-yr | Still real; multi-week.  Defer until B is tested. |
./docs/ocean/experiments/density_jacobian_pgf_mpas.md:1111:| (D) ico-5 | ~85 % | **REFUTED at available ETOPO.**  3× faster mode growth from finer-mesh dynamics + relocation.  Would also require ETOPO at much higher resolution AND a dramatic restructuring of the viscosity profile.  Not the production answer at this step. |
./docs/ocean/experiments/density_jacobian_pgf_mpas.md-1112-
./docs/ocean/experiments/density_jacobian_pgf_mpas.md-1113-### Strengthened case for Option B
./docs/ocean/experiments/density_jacobian_pgf_mpas.md-1114-
./docs/ocean/experiments/density_jacobian_pgf_mpas.md-1115-The §8e/§8f reading was that ρ_ref(z) was a band-aid attacking the
./docs/ocean/experiments/density_jacobian_pgf_mpas.md-1116-symptom; the dycore-expert demoted it because no production OGCM
--
./tests/ocean/unit/test_veros_global_1deg_recipe.py-142-
./tests/ocean/unit/test_veros_global_1deg_recipe.py-143-
./tests/ocean/unit/test_veros_global_1deg_recipe.py-144-def test_mxl_choice_1_post_mixing_is_accepted_and_debt_safe():
./tests/ocean/unit/test_veros_global_1deg_recipe.py-145-    """The faithful tke_mxl_choice=1 config (post-mixing Veros surface
./tests/ocean/unit/test_veros_global_1deg_recipe.py-146-    correction) is now ACCEPTED by _validate_post_mixing_cfg — the
./tests/ocean/unit/test_veros_global_1deg_recipe.py:147:    distance-to-boundary cap (veros_mxl_choice1_boundary_cap) makes it
./tests/ocean/unit/test_veros_global_1deg_recipe.py-148-    debt-safe, so no fallback is needed. Both choices validate; only an
./tests/ocean/unit/test_veros_global_1deg_recipe.py-149-    UNKNOWN choice raises (in compute_mixing_lengths, not the timing
./tests/ocean/unit/test_veros_global_1deg_recipe.py-150-    guard)."""
./tests/ocean/unit/test_veros_global_1deg_recipe.py-151-    from legoesm.ocean.physics.vertical_mixing.tke import (
./tests/ocean/unit/test_veros_global_1deg_recipe.py-152-        _validate_post_mixing_cfg,
--
./tests/ocean/unit/test_veros_global_1deg_recipe.py-161-
./tests/ocean/unit/test_veros_global_1deg_recipe.py-162-
./tests/ocean/unit/test_veros_global_1deg_recipe.py-163-def test_mxl_choice1_boundary_cap_bounds_the_surface_blowup():
./tests/ocean/unit/test_veros_global_1deg_recipe.py-164-    """veros_mxl_choice1_boundary_cap reproduces Veros tke.py:43-47: the
./tests/ocean/unit/test_veros_global_1deg_recipe.py-165-    raw buoyancy length overflows where N²→0 at the surface, and the cap
./tests/ocean/unit/test_veros_global_1deg_recipe.py:166:    clamps it to the distance-to-surface (a few metres). Non-vacuous: the
./tests/ocean/unit/test_veros_global_1deg_recipe.py-167-    UNCAPPED length is orders of magnitude larger."""
./tests/ocean/unit/test_veros_global_1deg_recipe.py-168-    import jax.numpy as jnp
./tests/ocean/unit/test_veros_global_1deg_recipe.py-169-    import numpy as np
./tests/ocean/unit/test_veros_global_1deg_recipe.py-170-    from legoesm.ocean.physics.vertical_mixing.tke import (
./tests/ocean/unit/test_veros_global_1deg_recipe.py-171-        veros_mxl_choice1_boundary_cap, compute_mixing_lengths,
--
./tests/ocean/unit/test_veros_global_1deg_recipe.py-498-
./tests/ocean/unit/test_veros_global_1deg_recipe.py-499-@pytest.mark.slow
./tests/ocean/unit/test_veros_global_1deg_recipe.py-500-def test_recipe_steps_once_finite_synthetic_with_solar():
./tests/ocean/unit/test_veros_global_1deg_recipe.py-501-    """Full stack + the q_solar column steps once finite at the STOCK
./tests/ocean/unit/test_veros_global_1deg_recipe.py-502-    360x160x115 size (slow: one jitted step of the full model).  Runs the
./tests/ocean/unit/test_veros_global_1deg_recipe.py:503:    FAITHFUL tke_mxl_choice=1 (no fallback) — the distance-to-boundary cap
./tests/ocean/unit/test_veros_global_1deg_recipe.py-504-    makes it debt-safe (see test_mxl_choice1_boundary_cap_bounds_the_surface
./tests/ocean/unit/test_veros_global_1deg_recipe.py-505-    _blowup + the recipe docstring)."""
./tests/ocean/unit/test_veros_global_1deg_recipe.py-506-    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
./tests/ocean/unit/test_veros_global_1deg_recipe.py-507-        LatLonCGridOceanModel,
./tests/ocean/unit/test_veros_global_1deg_recipe.py-508-    )
--
./tests/parallel/test_warmup_tiled_cube_comms.py-1-"""Deterministic NCCL comm-init warmup for the closed-loop tiled cube (#921).
./tests/parallel/test_warmup_tiled_cube_comms.py-2-
./tests/parallel/test_warmup_tiled_cube_comms.py-3-The closed-loop blocked step fuses, in ONE executable, the halo
./tests/parallel/test_warmup_tiled_cube_comms.py:4:collective-permute cliques AND the mass-fixer ``psum`` all-reduce clique over
./tests/parallel/test_warmup_tiled_cube_comms.py-5-the SAME ``(face, tile_i, tile_j)`` axes.  On multi-process GPU the two clique
./tests/parallel/test_warmup_tiled_cube_comms.py-6-KINDS can be scheduled for NCCL comm-init in a different relative order per rank
./tests/parallel/test_warmup_tiled_cube_comms.py-7-and deadlock; :func:`warmup_tiled_cube_comms` breaks that by priming each clique
./tests/parallel/test_warmup_tiled_cube_comms.py-8-kind in isolation, in a fixed order, before the first real step.
./tests/parallel/test_warmup_tiled_cube_comms.py-9-
--
./tests/parallel/test_warmup_tiled_cube_comms.py-65-    with pytest.raises(ValueError, match=r"\(6, kt, kt\)"):
./tests/parallel/test_warmup_tiled_cube_comms.py-66-        warmup_tiled_cube_comms(bad, KT, force=True)
./tests/parallel/test_warmup_tiled_cube_comms.py-67-
./tests/parallel/test_warmup_tiled_cube_comms.py-68-
./tests/parallel/test_warmup_tiled_cube_comms.py-69-def test_warmup_executables_split_the_two_clique_kinds():
./tests/parallel/test_warmup_tiled_cube_comms.py:70:    """The halo executable issues ONLY collective-permutes; the reduce
./tests/parallel/test_warmup_tiled_cube_comms.py-71-    executable issues ONLY an all-reduce.  Each NCCL comm therefore inits in
./tests/parallel/test_warmup_tiled_cube_comms.py-72-    isolation (the whole point) — replicated here with the SAME perms/axes the
./tests/parallel/test_warmup_tiled_cube_comms.py-73-    warmup uses."""
./tests/parallel/test_warmup_tiled_cube_comms.py-74-    mesh = _mesh()
./tests/parallel/test_warmup_tiled_cube_comms.py-75-    perms = (list(get_tiled_tables(KT).perms)
--
./tests/parallel/test_warmup_tiled_cube_comms.py-95-    halo = jax.jit(_halo_warm).lower(dummy).compile().as_text()
./tests/parallel/test_warmup_tiled_cube_comms.py-96-    red = jax.jit(_reduce_warm).lower(dummy).compile().as_text()
./tests/parallel/test_warmup_tiled_cube_comms.py-97-
./tests/parallel/test_warmup_tiled_cube_comms.py-98-    def _cp(h):
./tests/parallel/test_warmup_tiled_cube_comms.py-99-        return sum(1 for ln in h.splitlines()
./tests/parallel/test_warmup_tiled_cube_comms.py:100:                   if "collective-permute" in ln and "done" not in ln)
./tests/parallel/test_warmup_tiled_cube_comms.py-101-
./tests/parallel/test_warmup_tiled_cube_comms.py-102-    assert _cp(halo) > 0 and "all-reduce" not in halo
./tests/parallel/test_warmup_tiled_cube_comms.py-103-    assert "all-reduce" in red and _cp(red) == 0
--
./docs/ocean/experiments/silvestri_weno_reproduction.md-92-baroclinic_adjustment gate (48×48×8, dt=600, weno9, vs the Oceananigans oracle), the two solvers are
./docs/ocean/experiments/silvestri_weno_reproduction.md-93-**within ~3% of each other and both over-energize ~1.3× vs the oracle** (day-18 surface max|u|:
./docs/ocean/experiments/silvestri_weno_reproduction.md-94-oracle 1.56; `implicit_cn` 1.96 = 1.26×; `explicit_substep` 2.03 = 1.30×), neither blows up.
./docs/ocean/experiments/silvestri_weno_reproduction.md-95-⇒ **the free-surface/barotropic coupling is NOT the §5 residual lever** — switching to the
./docs/ocean/experiments/silvestri_weno_reproduction.md-96-oracle-faithful `ImplicitFreeSurface` (`implicit_cn`) does NOT reduce the over-energization. This
./docs/ocean/experiments/silvestri_weno_reproduction.md:97:INDEPENDENTLY CONFIRMS "the blowup is INSUFFICIENT DISSIPATION" (above): both solvers leak the same
./docs/ocean/experiments/silvestri_weno_reproduction.md-98-~1.3× excess energy, so the lever is grid-scale momentum dissipation (the wall representation +
./docs/ocean/experiments/silvestri_weno_reproduction.md-99-the anti-dissipative WENO pre-filter), exactly what **PR #559** (remove the `point_to_cellavg`
./docs/ocean/experiments/silvestri_weno_reproduction.md-100-anti-dissipative pre-filter) targets. internal_tide's `Flat`-y cure does NOT transfer (§5 is a full
./docs/ocean/experiments/silvestri_weno_reproduction.md-101-y-walled eddy field, not a y-uniform 2-D case). Reproduce:
./docs/ocean/experiments/silvestri_weno_reproduction.md-102-`BARO_SOLVER=implicit_cn|explicit_substep .venv/bin/python
--
./tests/unit/test_jra55_do.py-82-    half = 90.0 / n_lat
./tests/unit/test_jra55_do.py-83-    lat = np.linspace(90.0 - half, -90.0 + half, n_lat)   # descending
./tests/unit/test_jra55_do.py-84-    half_lon = 180.0 / n_lon
./tests/unit/test_jra55_do.py-85-    lon = np.linspace(half_lon, 360.0 - half_lon, n_lon)
./tests/unit/test_jra55_do.py-86-
./tests/unit/test_jra55_do.py:87:    # Synthetic field generators — keep values inside _PLAUSIBLE_RANGE.
./tests/unit/test_jra55_do.py-88-    def _scaled(low, high):
./tests/unit/test_jra55_do.py-89-        rng = np.random.default_rng(int((low * 1000 + high) % 2**31))
./tests/unit/test_jra55_do.py-90-        return rng.uniform(low, high, size=(n_records, n_lat, n_lon))
./tests/unit/test_jra55_do.py-91-
./tests/unit/test_jra55_do.py-92-    data_vars = {
--
./docs/physics-notes/pseudo_incompressible_les.md-209-review (mirrors serial, parity-gated).
./docs/physics-notes/pseudo_incompressible_les.md-210-SCALING: architecturally enabled (nearest-neighbour halo + O(1) allreduce/dot) + parity-proven;
./docs/physics-notes/pseudo_incompressible_les.md-211-a clean weak/strong SWEEP needs a real multi-node cluster (eager loop + mpi4jax per-collective
./docs/physics-notes/pseudo_incompressible_les.md-212-logging + 2 laptop cores → absolute timings noise-dominated here). Use repo docs/scaling infra.
./docs/physics-notes/pseudo_incompressible_les.md-213-
./docs/physics-notes/pseudo_incompressible_les.md:214:## GPU full-physics (2026-06-19): CONFIRMED stable
./docs/physics-notes/pseudo_incompressible_les.md-215-RTX 5090, full BL physics (vreman + flux/most_cooling + Coriolis): convective BL (surface
./docs/physics-notes/pseudo_incompressible_les.md-216-heating) 64³ 500 steps STABLE fp32 AND fp64 (no NaN, θ 300→306, w_max≈1.1). fp64 59/fp32 16
./docs/physics-notes/pseudo_incompressible_les.md-217-ms/step. Fine-res GABLS1-COOLING long run blows up (stable-BL grid-noise collapse) — added
./docs/physics-notes/pseudo_incompressible_les.md-218-composable `cfg.nu_floor` (spectral-core-proven; 14 tests still pass) but it ALONE is
./docs/physics-notes/pseudo_incompressible_les.md-219-insufficient at 64³ dx=6.25; needs an FD de-noising filter/hyperdiff (spectral core uses a
--
./tests/parallel/test_ppermute_halo_exchange.py-86-    assert got.shape == ref.shape == (6, n + 2, n + 2, nlev)
./tests/parallel/test_ppermute_halo_exchange.py-87-    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
./tests/parallel/test_ppermute_halo_exchange.py-88-
./tests/parallel/test_ppermute_halo_exchange.py-89-
./tests/parallel/test_ppermute_halo_exchange.py-90-def test_routing_emits_the_requested_collective():
./tests/parallel/test_ppermute_halo_exchange.py:91:    """The HLO must actually contain collective-permute for ppermute and
./tests/parallel/test_ppermute_halo_exchange.py-92-    all-gather for all_gather (codex: the value test alone could pass even if it
./tests/parallel/test_ppermute_halo_exchange.py-93-    silently fell back)."""
./tests/parallel/test_ppermute_halo_exchange.py-94-    n = 8
./tests/parallel/test_ppermute_halo_exchange.py-95-    data = jax.random.normal(jax.random.PRNGKey(5), (6, n, n))
./tests/parallel/test_ppermute_halo_exchange.py-96-    mesh = _mesh()
./tests/parallel/test_ppermute_halo_exchange.py-97-    cx.set_ppermute_default(True)
./tests/parallel/test_ppermute_halo_exchange.py-98-    hlo_pp = jax.jit(lambda d: cx.explicit_pad_halo(d, mesh, halo=1)).lower(
./tests/parallel/test_ppermute_halo_exchange.py-99-        data).compile().as_text()          # COMPILED HLO surfaces the collective
./tests/parallel/test_ppermute_halo_exchange.py:100:    assert "collective-permute" in hlo_pp and "all-gather" not in hlo_pp
./tests/parallel/test_ppermute_halo_exchange.py-101-    cx._cache.clear()
./tests/parallel/test_ppermute_halo_exchange.py-102-    cx.set_ppermute_default(False)
./tests/parallel/test_ppermute_halo_exchange.py-103-    hlo_ag = jax.jit(lambda d: cx.explicit_pad_halo(d, mesh, halo=1)).lower(
./tests/parallel/test_ppermute_halo_exchange.py-104-        data).compile().as_text()
./tests/parallel/test_ppermute_halo_exchange.py-105-    assert "all-gather" in hlo_ag
--
./tests/parallel/test_ppermute_halo_exchange.py-123-
./tests/parallel/test_ppermute_halo_exchange.py-124-
./tests/parallel/test_ppermute_halo_exchange.py-125-def test_halo2_routes_ppermute_multiface_and_matches():
./tests/parallel/test_ppermute_halo_exchange.py-126-    """halo=2 with the ppermute default ON routes to the multiface
./tests/parallel/test_ppermute_halo_exchange.py-127-    ppermute kernel (k=1 at 6 devices) — it must bit-match serial AND
./tests/parallel/test_ppermute_halo_exchange.py:128:    actually emit collective-permute, not the retired all_gather
./tests/parallel/test_ppermute_halo_exchange.py-129-    fallback (the pre-multiface code hard-routed halo=2 to
./tests/parallel/test_ppermute_halo_exchange.py-130-    allgather_h2; see tests/parallel/test_ppermute_multiface.py for
./tests/parallel/test_ppermute_halo_exchange.py-131-    the full multiface matrix)."""
./tests/parallel/test_ppermute_halo_exchange.py-132-    n = 8
./tests/parallel/test_ppermute_halo_exchange.py-133-    data = jax.random.normal(jax.random.PRNGKey(4), (6, n, n))
--
./tests/parallel/test_ppermute_halo_exchange.py-138-    got = cx.explicit_pad_halo(data, mesh, halo=2)
./tests/parallel/test_ppermute_halo_exchange.py-139-    assert got.shape == ref.shape == (6, n + 4, n + 4)
./tests/parallel/test_ppermute_halo_exchange.py-140-    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
./tests/parallel/test_ppermute_halo_exchange.py-141-    hlo = jax.jit(lambda d: cx.explicit_pad_halo(d, mesh, halo=2)).lower(
./tests/parallel/test_ppermute_halo_exchange.py-142-        data).compile().as_text()
./tests/parallel/test_ppermute_halo_exchange.py:143:    assert "collective-permute" in hlo and "all-gather" not in hlo
./tests/parallel/test_ppermute_halo_exchange.py-144-
./tests/parallel/test_ppermute_halo_exchange.py-145-
./tests/parallel/test_ppermute_halo_exchange.py-146-def test_ppermute_schedule_covers_24_adjacencies_once():
./tests/parallel/test_ppermute_halo_exchange.py-147-    """Invariant the ppermute correctness relies on: the 4 rounds cover all 24
./tests/parallel/test_ppermute_halo_exchange.py-148-    directed face adjacencies exactly once, and each round is a permutation
--
./docs/dev-notes/ocean_faithfulness_nemo.md-155-but NOT transports — transports are EQUILIBRATION-GATED** (need multi-decade runs, compute-heavy, like
./docs/dev-notes/ocean_faithfulness_nemo.md-156-the cube ¼°). The AMOC diagnostic (`_amoc26n_diag` in-run + `scripts/validate/nemo_transports.py`) is
./docs/dev-notes/ocean_faithfulness_nemo.md-157-committed + correct; a meaningful AMOC match requires an equilibrated run. Multi-year MPAS run launched
./docs/dev-notes/ocean_faithfulness_nemo.md-158-to show AMOC DEVELOPMENT + multi-year stability. ACC@Drake is also wind-driven-but-multi-year — same gate.
./docs/dev-notes/ocean_faithfulness_nemo.md-159-
./docs/dev-notes/ocean_faithfulness_nemo.md:160:### CODEX adversarial review of the cube-¼° + mpas-#160 parked conclusions (iter-~51) — BOTH CONFIRMED
./docs/dev-notes/ocean_faithfulness_nemo.md-161-- **CUBE: confirm parked.** Codex (code-grounded): the cold-start gate is the BAROCLINIC partial-cell
./docs/dev-notes/ocean_faithfulness_nemo.md-162-  PGF residual (the stratified-rest-over-bathy test has exact zero horizontal PGF yet blows at the
./docs/dev-notes/ocean_faithfulness_nemo.md-163-  bottom level ~step 180 → the motion is generated in the 3D `dp_dx/rho_0` path, ocean_pe_cdgrid:566).
./docs/dev-notes/ocean_faithfulness_nemo.md-164-  The barotropic solver only sees `eta`/`g∇η`; an implicit-CN barotropic damps gravity-wave CFL + null
./docs/dev-notes/ocean_faithfulness_nemo.md-165-  modes but CANNOT remove the depth-local baroclinic PGF acceleration injected every PE step → the
--
./docs/dev-notes/ocean_faithfulness_nemo.md-169-- **mpas #160: confirm parked.** The relative-PV + split-Matsuno-Coriolis is internally consistent,
./docs/dev-notes/ocean_faithfulness_nemo.md-170-  stable, matches NEMO SST/SSS. #160 (full PV q=(f+ζ)/h) is a TRiSK energy/enstrophy-invariant /
./docs/dev-notes/ocean_faithfulness_nemo.md-171-  elegance refactor with NO demonstrated SST/SSS-fidelity payoff; park unless a dynamic metric (energy
./docs/dev-notes/ocean_faithfulness_nemo.md-172-  drift, barotropic-Rossby phase, near-inertial spectrum) fails.
./docs/dev-notes/ocean_faithfulness_nemo.md-173-
./docs/dev-notes/ocean_faithfulness_nemo.md:174:### PGF-ZERO FALSIFICATION (8426611) — REFUTES "PGF is the SOLE cause"; a 2nd mode exists
./docs/dev-notes/ocean_faithfulness_nemo.md-175-Cube rest+topo+no-forcing+RK3, C32: **smc03 control blows step 210** (Med lev14, known); **zero-PGF
./docs/dev-notes/ocean_faithfulness_nemo.md-176-(`--cube-pgf-scheme zero`, ALL pressure force removed) STILL blows — delayed to step ~420, seed moves
./docs/dev-notes/ocean_faithfulness_nemo.md-177-to 20.9N/277E lev17.** A rest state with NO PGF has NO horizontal momentum force (Coriolis·v / adv·u /
./docs/dev-notes/ocean_faithfulness_nemo.md-178-KE all vanish from rest) yet still blows ⇒ **a SECOND cold-start instability source exists, independent
./docs/dev-notes/ocean_faithfulness_nemo.md-179-of the baroclinic PGF** (slower; candidate = the barotropic `g∇η` / partial-cell bathy / eta-floor
--
./docs/dev-notes/ocean_faithfulness_nemo.md-212---cube-pgf-scheme zero (all gated/diagnostic; faithful paths bit-unchanged).
./docs/dev-notes/ocean_faithfulness_nemo.md-213-
./docs/dev-notes/ocean_faithfulness_nemo.md-214-### C256 ¼° cube probe (8427650) — mode-1 is NOT resolution-fixable
./docs/dev-notes/ocean_faithfulness_nemo.md-215-rest+topo+no-forcing, FULL corrected stack (smc03 + 2nd-order bottom slope + bathy-smooth5 + RK3),
./docs/dev-notes/ocean_faithfulness_nemo.md-216-C256 (¼°)/nlev20/dt10: **blows step ~288**, seed `umax_lat 1.1, umax_lon 314.8, lev 13` = a **cube
./docs/dev-notes/ocean_faithfulness_nemo.md:217:face EDGE (lon 315) right at the equator**, mid-depth, |u| 400→508 m/s in 6 steps. This REFUTES the
./docs/dev-notes/ocean_faithfulness_nemo.md-218-prior "needs ¼° + full stack" hope: ¼° does NOT clear mode-1. Sharper diagnosis — the residual
./docs/dev-notes/ocean_faithfulness_nemo.md-219-concentrates where (a) the AL corner gradient crosses a cube face seam AND (b) f→0 removes the
./docs/dev-notes/ocean_faithfulness_nemo.md-220-geostrophic restraint, so any spurious/real baroclinic PGF accelerates unchecked. The 3 implicit-CN
./docs/dev-notes/ocean_faithfulness_nemo.md-221-grids (tripole/latlon/MPAS) ride through the same WOA cold-start because their unconditionally-stable
./docs/dev-notes/ocean_faithfulness_nemo.md-222-barotropic absorbs the fast equatorial adjustment; the cube's EXPLICIT fv3sw cannot. **Cube cold-start
--
./docs/dev-notes/ocean_faithfulness_nemo.md-407-- **SSS 1.42→1.38** (bias −0.47→−0.44): marginal-sea restoring helps MODESTLY — the continental extremes
./docs/dev-notes/ocean_faithfulness_nemo.md-408-  (Baltic/Hudson/Okhotsk) are reduced but NOT eliminated (the NW-Pacific/Japan-Sea patch persists, partly
./docs/dev-notes/ocean_faithfulness_nemo.md-409-  south of the Okhotsk region; enclosed-strait dynamics unresolved at ~115 km). Partial fix, not complete.
./docs/dev-notes/ocean_faithfulness_nemo.md-410-- SST 1.57/0.989 (unchanged), ACC 133.7.
./docs/dev-notes/ocean_faithfulness_nemo.md-411-- **CORRECTED AMOC (section) = 6.05 Sv** (vs the inflated band-sum 13.3; NEMO 17.7) — the REAL τ=60 AMOC is
./docs/dev-notes/ocean_faithfulness_nemo.md:412:  LOW. CONFIRMS the SSS↔AMOC trade-off is REAL + the E−P decoupling is only PARTIAL: strong restoring holds
./docs/dev-notes/ocean_faithfulness_nemo.md-413-  SSS (1.38) but genuinely WEAKENS the AMOC (~6). τ=365 (better AMOC, ~13 est) has bad SSS. No single τ wins
./docs/dev-notes/ocean_faithfulness_nemo.md-414-  on both. MHT 3.3 PW (section; was 5-6 band-sum; still ~1.8× obs ~1.8 — possible real over-transport OR a
./docs/dev-notes/ocean_faithfulness_nemo.md-415-  residual per-latitude-crossing subtlety; secondary, flag).
./docs/dev-notes/ocean_faithfulness_nemo.md-416-- HIGHER-RES (the SST-gap lever) in flight on glab1: ico7-2yr (day-90 stable, ~50h; year-1 read ~10h),
./docs/dev-notes/ocean_faithfulness_nemo.md-417-  latlon-0.5° (slow host stepping ~28h, no blowup yet). ico7-90day on a bad short node was step-0-stuck →
--
./docs/dev-notes/ocean_faithfulness_nemo.md-542-  gradient/flux terms, amplified by the scorer's cKDTree-IDW blend across the seam.
./docs/dev-notes/ocean_faithfulness_nemo.md-543-- **PRE-EXISTING, not from this session's flux work:** regridded lon-72 stripe sharpness is 2.71°C in the OLD
./docs/dev-notes/ocean_faithfulness_nemo.md-544-  ice-thermo tripole vs 2.86°C in the NCAR run — essentially identical. The latlon 1° grid has NO such seam by
./docs/dev-notes/ocean_faithfulness_nemo.md-545-  construction (yesterday's clean map). The physical domain i=1..360 integrates correctly (tripole is stable +
./docs/dev-notes/ocean_faithfulness_nemo.md-546-  SST-faithful); only the 2 halo columns are off, so the seam is mild not catastrophic.
./docs/dev-notes/ocean_faithfulness_nemo.md:547:- **ROOT CAUSE CONFIRMED at mesh level:** the eORCA1 mesh_mask's cyclic halo columns i=0/i=361 are marked
./docs/dev-notes/ocean_faithfulness_nemo.md-548-  LAND everywhere (0 wet) while their ORCA-overlap partners i=360/i=1 are ocean (147/143 wet) — 143 latitudes
./docs/dev-notes/ocean_faithfulness_nemo.md-549-  where i=1 is ocean but its west-neighbour i=0 is a fake land wall. NEMO fills these halos every step via
./docs/dev-notes/ocean_faithfulness_nemo.md-550-  `lbc_lnk`; legoESM read `tmaskutil` raw and never applied the overlap → the lon-72.5 seam ocean is severed.
./docs/dev-notes/ocean_faithfulness_nemo.md-551-  DEEPER: legoESM treats the (332,362) grid as a period-**362** ring (operators `jnp.roll(...,axis=1)` over
./docs/dev-notes/ocean_faithfulness_nemo.md-552-  all 362 cols), but eORCA1 is physically period-**360** with 2 duplicate-longitude overlap halos — so even
--
./docs/dev-notes/ocean_faithfulness_nemo.md-735-clear day-270 (where full-stack+ppm diverged) to confirm; it has ~15h walltime, reaches 2yr ~day-730.
./docs/dev-notes/ocean_faithfulness_nemo.md-736-**Production plan crystallizing:** `--tracer-advection tvd` (NOT ppm_fct) + `--visc-schedule
./docs/dev-notes/ocean_faithfulness_nemo.md-737-0:1e5:0.33,90:5e4:0.33,180:2e4:0.33` + `--sw-rgb-chl` + `--sss-ice-gate-nemo` + seam + runoff-spread +
./docs/dev-notes/ocean_faithfulness_nemo.md-738-river-gate + BBL. Launch once full_tvd passes day-270.
./docs/dev-notes/ocean_faithfulness_nemo.md-739-
./docs/dev-notes/ocean_faithfulness_nemo.md:740:### iter-I BISECT VERDICT — ppm_fct CONFIRMED as the day-360 NaN driver (2026-06-11)
./docs/dev-notes/ocean_faithfulness_nemo.md-741-**full_tvd 8459834 at day-270: max|u| 0.78 m/s, STABLE.** The smoking gun: the original full-stack+ppm
./docs/dev-notes/ocean_faithfulness_nemo.md-742-run (8458811) hit max|u| **3.15 at the SAME day-270** (its NaN precursor → NaN day-360). Same stack,
./docs/dev-notes/ocean_faithfulness_nemo.md-743-only the tracer advection differs (tvd vs ppm_fct) → tvd is **4× calmer** at the exact divergence point.
./docs/dev-notes/ocean_faithfulness_nemo.md-744-ppm_fct's front-sharpening drove the velocity blowup; **tvd is the fix.** Full trajectory: tvd day-90
./docs/dev-notes/ocean_faithfulness_nemo.md-745-0.71 / day-180 1.05 / day-270 0.78 — bounded, no runaway (ppm: day-270 3.15 → NaN day-360). day-360
--
./docs/dev-notes/clubb_port_history.md-70-|-------|----------|-------|
./docs/dev-notes/clubb_port_history.md-71-| C1 | clubb_diagnostic, clubb_core | |
./docs/dev-notes/clubb_port_history.md-72-| C2 | clubb_coefficients, clubb_wp23, clubb_xm_wpxp | coefficients co-absorbed to break its `compute_skw_fnc` edge cycle-free; `_GAMMA` deduped |
./docs/dev-notes/clubb_port_history.md-73-| C3 | clubb_moments, clubb_mfl | `_EPS` deduped; MFL's NaN-propagating `_safe_sqrt` kept local; no-`__all__` policy documented |
./docs/dev-notes/clubb_port_history.md-74-| C4 | clubb_solve, clubb_fill_holes, clubb_skewness, clubb_tau, clubb_pdf, clubb_pdf_moments | 6 constants deduped; `_F64_EPS` float()-unified (bool-only uses) |
./docs/dev-notes/clubb_port_history.md:75:| C5 | clubb_grid, clubb_saturation, clubb_helpers, clubb_mixing_length | saturation = thin adapters over shared thermo Flatau curves; mixing length CONFIRMED different from `_shared`'s Blackadar |
./docs/dev-notes/clubb_port_history.md-76-| C6 | clubb_config | `CLUBBFlags` class REMOVED: 14 flag sites hardcoded at CAM values, dead non-CAM branches deleted, 65 flag values → machine-parseable end-of-file table with namelist tripwire preserved |
./docs/dev-notes/clubb_port_history.md-77-| C7 | — | `clubb_lite.py` revert (inline g/θ_v restored byte-identical to main; formula-budget baseline restored); rebase-fallout budget re-seeds |
./docs/dev-notes/clubb_port_history.md-78-| C8 | — | line-numbered TOC + `test_toc_line_numbers_accurate` drift tripwire |
./docs/dev-notes/clubb_port_history.md-79-| C9 | — | whole-repo gate sweep (see below) |
./docs/dev-notes/clubb_port_history.md-80-
--
./docs/dev-notes/CRM_faithful_SAM.md-42-cross-check. The iter-68/174 detail below is retained as history.
./docs/dev-notes/CRM_faithful_SAM.md-43-
./docs/dev-notes/CRM_faithful_SAM.md-44-**STATUS (iter-68, codex judgment review b8u2md64l — TONED DOWN from overclaim):** Component
./docs/dev-notes/CRM_faithful_SAM.md-45-implementations are SOURCE-FAITHFUL where checked (each codex-reviewed). RCE produces SAM/RCEMIP-like
./docs/dev-notes/CRM_faithful_SAM.md-46-bulk THERMODYNAMIC + convective-magnitude profiles. **REMAINING UNVALIDATED RISKS (codex, honest):**
./docs/dev-notes/CRM_faithful_SAM.md:47:(1) **momentum-advection sensitivity / convective EXTREMES** — CONFIRMED: van_leer (2nd-TVD, default)
./docs/dev-notes/CRM_faithful_SAM.md-48-SUPPRESSES updraft cores — identical-IC max|w| caps ~2.6 m/s vs weno5 (low-diffusion) 14.8/5.3/3.3;
./docs/dev-notes/CRM_faithful_SAM.md-49-bulk w_RMS means match but the TAILS don't. SAM uses NON-diffusive 2nd-order CENTERED momentum
./docs/dev-notes/CRM_faithful_SAM.md-50-(`advect2_mom_xy.f90`) ⇒ **per-field split (centered/low-diff momentum + monotone scalars) is MOTIVATED,
./docs/dev-notes/CRM_faithful_SAM.md-51-not optional — REOPENED #86**; (2) equilibrium cloud/ANVIL structure (long rrtmgp run in flight);
./docs/dev-notes/CRM_faithful_SAM.md-52-(3) case-specific GATE/LBA TRIGGERING/organization — only smoke+config-faithful, NOT convectively
--
./docs/dev-notes/CRM_faithful_SAM.md-201-## Iteration history (compact)
./docs/dev-notes/CRM_faithful_SAM.md-202-| iter | change |
./docs/dev-notes/CRM_faithful_SAM.md-203-|---|---|
./docs/dev-notes/CRM_faithful_SAM.md-204-| 1-50 | Faithful components built+audited: SMAG, D1-D6 dynamics, Large-Pond surface flux, RAD-1..8 (rrtmgp), full M2005 2-moment (warm+ice+snow+graupel), GATE/LBA/RCE drivers, comparison diagnostics; periodic doc shrinks. (Substance → Findings section above.) |
./docs/dev-notes/CRM_faithful_SAM.md-205-| 51-60 | **VGRID** exact SAM grd reader (266 lvl). **HYPERDIFF** audit (SAM none; ours weak dx-aware ∇⁴ for the acoustic 2Δx). **FORCING-T** real bug (lsf `tls`=absolute-dT/dt not dθ/dt ⇒ /exner_ref fix; was GATE warm bias; exner lone-miss proven). GATE/RCE forcing+IC audits FAITHFUL; **SND-RH** reject RH/NaN snd; gray→rrtmgp default. **SND-TOP** US-std-atm T-ratio stratosphere (was 83 K neutral-clamp). **SGS-SCALAR** diffuse all tracers w/ K_h (was θ'-only). **SGS-VERT** gap found (#81). |
./docs/dev-notes/CRM_faithful_SAM.md:206:| 61-66 | **GPU runs begin** (RTX 5090; SAM unbuildable ⇒ source-faithfulness only). **#82** blow-up FIXED (root: `substep_horizontal_acoustic=False` default; flipped ON; ~8 hypotheses disproven). **#83** no-convection FIXED (IC θ-profile bug: Γ=6.7 K/km used as θ-grad not T-lapse ⇒ ~0 CAPE; delegated to library Wing profile) ⇒ **RCE DEEP-CONVECTS, matches RCEMIP-SAM** (CWV 47-48 mm, mid-trop w_RMS 0.41-0.48, mid-trop-peaked w'², cloud 0.2; +codex #83 review: q_sfc-consistency + conditional-instability guard tests). **#85** root = gray under-driving ⇒ gray→rrtmgp default (hyperdiff REFUTED as cause via A/B). **#81 SGS-VERT** now fully 3D = SAM's exact explicit operator, **mass-weighted** `(1/ρ)∂_z(ρ_w K∂_zφ)` (codex [HIGH]); A/B: smooths burst variance ~25% + cuts tropopause θ'-accum ~33% (more SAM-like). **#75 FORCING-SUB** investigated = NOT a bug. **#86** advection: identical-IC weno5-vs-van_leer A/B ⇒ van_leer adequate for time-mean magnitudes (codex [S1] resolved; buoyancy verified faithful to `buoyancy.f90`). |
./docs/dev-notes/CRM_faithful_SAM.md-207-| 67-71 | GATE rrtmgp UNBLOCKED via SAM `nrad`-cadence cached radiation (90 s period, 45-90× fewer rrtmgp calls; `--radiation-interval`). **codex JUDGMENT review (b8u2md64l) corrected the iter-66 advection overclaim** → implemented+validated 2nd-order CENTERED-momentum split (=gSAM `advect2_mom`; opt-in `--momentum-advection centered`, stable to a real ~8 m/s RRTM burst; KE-budget quantifies van_leer ≈ −0.4‖u‖² dissipative vs centered ≈ +0.015 near-conserving). **VERIFIED explicit vertical SGS = SAM's EXACT spatial operator** for every loop case (`diffuse_scalar3D.f90` `doimplicitdiff=.false.` leg; implicit tridiag is global-presets-only). BONUS: fixed a pre-existing #80 tracer-SGS serial≠MPI divergence. GATE `--emit-profiles` tooling. Doc shrink (239→207). **#88: GATE non-convection = FAITHFUL CIN** (CAPE=1205 J/kg realized — RCE proves the model convects given CAPE; CIN-capped, eroded only over ~10-25 h forcing = compute-bound; even a 2 K seed can't punch the CIN). |
./docs/dev-notes/CRM_faithful_SAM.md-208-| 72-75 | **M2005 latent-heating → θ' → buoyancy coupling VERIFIED end-to-end** (codex holistic review b2dcri8e0). codex's [HIGH] decisive CIN-bypass test: a +3 K saturated bubble above the GATE LFC (z=1.06 km), microphysics-ONLY ⇒ **max\|w\| grows 0.18→5.37 m/s, condensate forms, θ' RISES via latent heat (3.0→3.58)** ⇒ the model REALIZES CAPE ⇒ GATE's non-convection is the FAITHFUL CIN (#88), NOT a lost-heating bug (code path verified: `morrison.py:797-977` builds dT_dt → `integration.py:420` `dθ'/dt=dT_dt/exner` → prognostic θ'). Locked in `tests/validation/test_moist_bubble_cape_realization.py`, then **fault-injection-VALIDATED** it catches the bug: zeroing the micro θ' tendency (keep condensation) → θ' DECAYS 3.0→2.28 < the 2.7 threshold (condensate+w alone don't discriminate — the q_v virtual buoyancy still lifts the bubble; the **θ' check is THE discriminator**). 29 plane-CRM tests green. codex caveat ACCEPTED: GATE/LBA at dx=1000 m vs SAM 100 m LES = coarse-CRM checks; **RCE (dx=4 km vs coarse RCEMIP-SAM) is the apples-to-apples rigorous case.** |
./docs/dev-notes/CRM_faithful_SAM.md-209-| 76-78 | **GATE/LBA convective-validation tooling + LBA case setup.** DRY: shared `run_rcemip_plane.emit_crm_profiles` (w'²/T/q_v±std/cloud/condensate/CWV/max\|w\|/precip bundle) wired into GATE+LBA `--emit-profiles` (removed GATE-local dup). Switched GPU GATE→LBA (the more-decisive 3rd case — strong land fluxes H=260/LE=536 W/m²; GATE onset is slow ~10-25 h CIN-erosion + fully characterized as faithful #88). **Caught+fixed an LBA diurnal-phase bug:** the `sfc` file spans sunrise day0.0 → midday peak 0.25 → sunset 0.493 and `surface_at_day` (`sam_case_forcing.py:503`) CLAMPS out-of-range days; a first run (day0=0.25→0.75) ran into the post-sunset clamped-ZERO-flux night (would've emitted decayed nocturnal profiles) ⇒ relaunched from sunrise day0=0.0. |
./docs/dev-notes/CRM_faithful_SAM.md-210-| 79-81 | **codex review (b3no4lcaf) of the LBA setup → addressed its [HIGH] + doc shrink.** codex [HIGH]: even the day0=0.0→0.44 run is FRAGILE — day 0.44 = 10.6 h post-sunrise, only 1.3 h pre-sunset, so the forcing already DECLINED (H+LE~174 vs the 796 midday peak) ⇒ an END snapshot catches the FALLING limb / cold-pool, not the convective peak. **FIXED: peak-tracking in `run_lba` — emits the PEAK-convection state's profiles (highest max\|w\| over the afternoon), not the end state**; re-ran day0=0.0→0.486 (full afternoon, pre-sunset), print-every 1000 (bhjy29ieq, results/lba_run3.log). codex other findings: [MED] smooth k=1 0.1 K seed = single-plume/trigger risk (band-limited random `_band_limited_seed_pattern` in rcp available if a single deterministic plume appears); [LOW] dt=2-vs-3 immaterial for magnitude; [MED] sfc-clamp OK for this sub-day run. **codex verdict: NO remaining proven physics-code fidelity bug; highest-value action = make LBA time-resolved (DONE via peak-tracking).** LBA smoke regression GREEN. Doc shrink: consolidated rows 72-78 (239→~207 net). |
./docs/dev-notes/CRM_faithful_SAM.md-211-| 82 | **Closed the documented C4-random-SEED faithfulness gap (codex [MED] + docs "minor" list): LBA now uses a SAM-faithful BAND-LIMITED RANDOM θ' seed (multi-cell), not a single smooth-k1 domain-scale plume.** Promoted `band_limited_seed_pattern` (modes 1≤\|k\|≤k_max — multi-cell WITHOUT grid-scale power that would NaN the acoustic IC; the SAM C4-noise analogue) from rcp → `sam_case_setup` (DRY: rcp now imports it); added `seed_kind` (`smooth_k1`/`band_random`, ValueError on unknown) to `build_sam_case_initial_state`; defaulted `build_lba_setup` to `band_random`. Re-ran LBA (24²×50, day0=0.0→0.486, **peak-tracking + random seed**, bwxhsre22) — now faithful (multi-cell) AND robust (peak-over-afternoon). Verified GREEN: LBA/GATE smoke, moist-bubble, + new DIRECT unit tests for the seed functions (`band_limited_seed_pattern`: zero-mean/unit-std/band-limited-spectrum/reproducible/multi-cell; `_random_band_theta_seed`: BL-levels-only/zero-mean). |
./docs/dev-notes/CRM_faithful_SAM.md-212-| 83-86 | Seed-function direct unit tests (above); confirmed cumulative changes clean across GATE/LBA/RCE (cross-grid RCE smoke). **LBA CAPE diagnostic (`parcel_profile_and_cape`): LBA IC CAPE=1626 J/kg (morning, conditionally unstable, lapse 5.77 K/km), heated to 3000-6500 J/kg by afternoon — STRONGLY unstable.** So the LBA run's flat max\|w\| through noon (step 10k, day 0.23, at the H=269/LE=552 flux peak) is the FAITHFUL CIN-limited pre-onset (= GATE #88 mechanism — model realizes CAPE per the bubble test, but the morning CIN caps parcels), NOT a bug; LBA's strong heating erodes the CIN through the afternoon. **RESULT: the LBA run did NOT trigger** — max\|w\| stayed flat then DECLINED (8.8e-4→4.8e-4) as the post-peak fluxes waned (day 0.26-0.32). So LBA, like GATE, is **CIN-limited AT COARSE RESOLUTION**: the BL thermals that punch the CIN up to the LFC are UNRESOLVED at dx=1000 m (SAM-LBA is a dx=100 m LES). The model realizes CAPE (bubble test) ⇒ this is the **RESOLUTION limitation codex flagged [HIGH], NOT a fidelity bug.** **KEY CONCLUSION (#88+): the CIN-capped cases (GATE, LBA) cannot self-trigger their convection at dx=1000 m; RCE — no CIN, radiatively-destabilized WHOLE column — DOES trigger and IS the apples-to-apples validated case.** **LBA forced-updraft magnitude CONSISTENT via a bubble test** (codex iter-88 [HIGH]: this validates that the model converts LFC-level CAPE→upward acceleration with correct latent heating — the CONDITIONAL forced-updraft — NOT the domain statistics): saturated bubble above the LBA LFC@1.32 km, micro-only ⇒ max\|w\| grows 0.19→6.06 m/s with condensate 0.006→1.81 g/kg + latent-heat-raised θ'; plausibly scales with CAPE (6.06/5.4=1.12 vs √-CAPE 1.16, thin/2-case). **HONEST SCOPE: RCE's FULL DOMAIN convective magnitude is faithful (matches RCEMIP-SAM — the rigorous apples-to-apples case); for GATE+LBA the bubble confirms the CAPE→updraft conversion + coupling, but the DOMAIN statistics (precip, w'²(z), cold pools, organization, triggering frequency) need the self-triggered run, which is RESOLUTION-limited at dx=1000 m vs SAM's dx=100 m LES (codex [HIGH]) — a documented caveat, not a fidelity bug.** |
./docs/dev-notes/CRM_faithful_SAM.md:213:| 88 | **codex review (bu0829cei) of the FINAL conclusion → all findings addressed; verdict: NO code-level fidelity correction justified.** [HIGH] bubble validates only the CONDITIONAL forced-updraft + CAPE→buoyancy coupling, not the GATE/LBA DOMAIN stats — reworded (above). [MED] CAPE-scaling thin (6.06/5.4=1.12 vs √-CAPE 1.16, 2 cases) — softened. [MED] resolution-vs-over-damping falsification — ADDRESSED: (a) the sponge is upper-level only (sam_rational base 0.6·H=12 km; BL is bottom ~1-2 km) so it CANNOT eat BL turbulence; (b) SGS = SAM's exact operator (Cs=0.19, no wall cap) so not over-aggressive; (c) RCE + the bubble SUSTAIN resolved convection ⇒ no over-damping of resolved motions; (d) BL thermals (~500 m = BL depth) are SUB-GRID at dx=1000 m ⇒ the flat BL is resolution, not damping. + DIRECT test CONFIRMED: at **dx=250 m** the BL thermals DEVELOP — max\|w\| rises to 0.029 m/s @ step 500, **~36× the dx=1000 m flat (8e-4)** — so the BL turbulence is RESOLUTION-DEPENDENT (present at dx=250 m, sub-grid at dx=1000 m), NOT model over-damping. **QUANTIFIED (codex's w_CIN-vs-w* diagnostic): LBA CIN=22 J/kg (w_CIN=√2CIN=6.7 m/s), theoretical convective velocity scale w*=(B0·zi)^⅓=2.4 m/s at peak heating — but the RESOLVED BL max\|w\| is ~8e-4 @ dx=1000 m = ~3000× BELOW w* (severely under-resolved), → 0.028 @ dx=250 m. So the LFC-lifting BL thermals are quantitatively under-resolved at dx=1000 m. Resolution diagnosis CLOSED; conclusion holds — NO code-level fidelity correction exists** (codex Q4: a forced-trigger dx=1000 m run is NOT a meaningful vs-SAM-LES comparison — correctly not pursued). |
./docs/dev-notes/CRM_faithful_SAM.md-214-| 89-174 | **Verification + maintenance; conclusion unchanged + codex-validated.** All uncommitted changes GREEN: `test_sam_case_setup.py` (19 tests incl. the band-limited-seed properties) + `test_moist_bubble_cape_realization.py` pass. Completed codex's **w_CIN-vs-w\* BL diagnostic** (folded into the #88 row above): LBA CIN=22 J/kg ⇒ w_CIN=6.7 m/s, convective scale w\*=2.4 m/s, resolved BL max\|w\| ~3000× below w\* @ dx=1000 m → 0.028 @ dx=250 m ⇒ quantitatively closes resolution-vs-over-damping (under-resolved BL thermals, NOT over-damping). dx=250 m LBA run completed clean (max\|w\|=0.022, mass-conserving). **Doc shrink** (compressed history rows 1-71; removed the stale iter-64 closing block). **No code-level fidelity correction remains.** |
./docs/dev-notes/CRM_faithful_SAM.md-215-| 175-176 | **CONCRETE reproducible RCE-vs-RCEMIP-300 K comparison built + honesty-hardened** (`scripts/validate/compare_rce_vs_rcemip_sam.py`, tested `compute_metrics`/`evaluate`, figure). Faithful single-GPU RCE (48²×50, dx=3 km, rrtmgp+M2005+Smag, band_noise seed, `--semi-implicit`, 3D-SGS) bursts (max\|w\|≈8.7) then sustains; TIME-AVERAGED over the convective window (NOT a snapshot). Self-adversarial-review (codex CLI sandbox-blocked, `bwrap RTM_NEWADDR`) hardened it: CWV→dry specific-humidity, cold-point bound 202→198 K, an EDGE-transparency column, "quasi-equilibrium"→"transient convective phase"; code checks PASS (pressure `p=p_ref(T/θ)^(1/κ)`/CWV/RH/ascending-z/units, no bug). **Honest result: 6/7 within the RCEMIP 300 K MODEL-SPREAD; w_RMS at lo-edge (0.25 — later FIXED by delta_max iter-181), cold-point ABOVE (documented θ'-accumulation) — consistent with the spread, NOT a tight equilibrium match.** Caught a launch bug (`--theta-noise-amp` default 0.0 ⇒ un-seeded RCE stays laminar). |
./docs/dev-notes/CRM_faithful_SAM.md-216-| 177 | **LES + DNS capability added (user request) — small lift, reuses the CRM diffusion operators.** CRM/LES/DNS share the SAME horizontal + vertical `∂(K∂φ)` operators; only the viscosity K_m differs. Added a `turbulence_closure` config MODE (`"smagorinsky"` \| `"molecular"` \| `"none"`) + `molecular_viscosity`/`molecular_prandtl` fields (+ `constants.prandtl_air=0.71`; `nu_air=1.5e-5` already existed). **LES** = the existing Smagorinsky at fine dx (Cs~0.15, wall-damping, 3-D SGS) — already reachable, now an explicit mode. **DNS** = the `"molecular"` closure: CONSTANT molecular ν (K_m=ν, K_h=ν/Pr), NO eddy model, full 3-D ν∇² via `--sgs-vertical`; needs dx~Kolmogorov. Wired into BOTH serial + MPI-halo paths (a constant K is trivially halo-consistent; serial=halo parity test added) + guarded in `validate_plane_config` (ValueError on unknown closure / molecular ν≤0 / Pr≤0). Driver flags `--turbulence-closure` + `--molecular-viscosity` in `run_rcemip_plane.py` (echoed in run metadata). **Tests** (`tests/validation/test_dns_les_closure.py` ×5 + 1 halo-parity in `test_plane_slow_tend_halo.py`): the molecular closure reproduces the ANALYTICAL viscous decay `du/dt=−ν k² u` for a zero-divergence shear (rel 5e-3); `closure="none"` leaves it undamped (so the decay IS the molecular closure); the vertical leg damps a `u'(z)` shear; the Smagorinsky CRM/LES path is unchanged (backward-compat). 34-test dycore/halo/cross-grid regression GREEN. **Mode recipes:** CRM = smagorinsky, Cs 0.19, dx 1-4 km; LES = smagorinsky, Cs 0.15, dx 10-100 m, --sgs-vertical + wall-damping; DNS = molecular, ν, dx~Kolmogorov, --sgs-vertical. Faithfulness note: SAM's GATE/LBA ARE dx=100 m LES (Smagorinsky) — so the LES mode is the SAME closure as the gSAM oracle at the SAME resolution; DNS is the no-model molecular limit below LES. |
./docs/dev-notes/CRM_faithful_SAM.md-217-| 178 | **Self-adversarial-review of the iter-177 LES/DNS code** (codex CLI still sandbox-blocked on this host — `bwrap RTM_NEWADDR`). VERIFIED `validate_plane_config` IS called in `PlaneCompressibleEulerModel.__init__` (line 2347) ⇒ the closure guards fire at model construction, NOT dead code. **Addressed [MED]: a `molecular`/DNS config without `sgs_vertical_diffusion` silently applied HORIZONTAL-only molecular viscosity (incomplete DNS) — added a `validate_plane_config` warning + test** (`test_dns_horizontal_only_warns`). Confirmed the vertical molecular leg is mass-weighted `(1/ρ)∂_z(ρν∂_z φ)` = the correct molecular momentum diffusion (μ=ρν; SAM's anelastic ρ0-weighting coincides with molecular μ). Documented caveats: (a) the DNS scalar leg uses one `molecular_prandtl` for heat + all tracers (Sc≈Pr≈0.7 air approximation); (b) the bottom BC is flux/free-slip (surface scheme), so DNS is currently for free-shear / box turbulence — wall-bounded DNS would need a no-slip BC. 6 DNS-closure tests + 1 halo-parity GREEN; no code-level fidelity bug found in the closure. |
./docs/dev-notes/CRM_faithful_SAM.md-218-| 179 | **LES preset made fully driver-reachable + end-to-end smokes for all 3 modes.** Self-review caught that `run_rcemip_plane.py` hardcoded `smagorinsky_wall_damping=False` (SAM-faithful CRM) with NO override ⇒ the documented LES preset (which needs the wall cap `ℓ_m=min(Cs·Δ,κz)`) was not producible via the driver. Added `--smag-wall-damping`/`--no-smag-wall-damping` (default False = CRM). **All three recipes now RUN end-to-end (driver smokes):** CRM (default, `--smag-cs 0.19`); **LES** `--smag-cs 0.15 --smag-wall-damping --sgs-vertical --dx 100` (closure=smagorinsky, finite, mass-conserving); **DNS** `--turbulence-closure molecular --molecular-viscosity ν --sgs-vertical` (finite). 7 closure tests (6 DNS + molecular halo-parity) GREEN. The iter-177-178 config-field + closure-branch changes are backward-compatible (all `CompressibleEulerConfig` construction is keyword; the closure gate is bit-identical for the default smagorinsky mode — verified by halo serial=MPI parity). LES/DNS capability now complete, driver-usable, tested, reviewed. |
./docs/dev-notes/CRM_faithful_SAM.md-219-| 180 | **gSAM reference-data due-diligence — re-examined the "no live comparison" premise.** Searched the full gSAM tree: only INPUT data exists — `RUNDATA/` holds the RRTMG/CAM optics `.nc`; `CASES/*.nc` are observed FORCING (e.g. `GATE/GATE3h.nc` = real-GATE 3-hourly T/q/u/v/omega/divT/divq/H + apparent-heating Q1/Q2); `SCRIPTS/LI/out.nc` is a limited W/QV series of unknown provenance; `DOC/` is just the user-guide PDF. **NO `OUT_STAT`/`.stat`/`bin3D` SAM output statistics for GATE_IDEAL/LBA/RCE** ⇒ confirms there is no bundled gSAM output to compare against, so faithfulness genuinely must be vs SAM SOURCE formulas (done, each codex-reviewed) + published RCEMIP (iter-175 artifact). The observed GATE Q1/Q2 in `GATE3h.nc` is the classic CRM-GATE target but matching it needs a self-triggered/forced or LES GATE ensemble — resolution/compute-bound (the documented caveat). Premise re-validated, not assumed. |
./docs/dev-notes/CRM_faithful_SAM.md-220-| 181-182 | **REAL SGS faithfulness BUG found + fixed via source audit: the missing SAM `delta_max` mixing-length cap.** Audited legoESM's Smagorinsky vs the ACTIVE SAM `SGS_TKE` (Build:21, NOT the inactive `_ORIG`/`.old`): SAM caps the HORIZONTAL spacing at **`δmax=1000 m`** (`sgs.f90:90`; `grd=(dz·adz·min(δ,dx·mu)·min(δ,dy·ady))^⅓`); legoESM used the UNCAPPED `Δ=(dx·dy·dz)^⅓` ⇒ for dx>1 km (RCE 3-4 km) K_m was ~4-6× too strong ⇒ SGS OVER-MIXED. **Fix:** `smagorinsky_delta_max` config (default 1000) threaded to the strain fn in serial + halo (parity preserved) + driver `--smag-delta-max`; unit-tested (≈0.16× at dx=4 km, bit-identical for dx≤1 km ⇒ GATE/LBA unchanged); 10 closure + 3 halo-parity GREEN. **The REST of the SGS matches the active scheme bit-for-bit** (`|S|²=2S_ijS_ij` physical strain, `tk=Cs²Δ²|S|` via Ck/Cee/Cs with Cs=0.19/Pr=1/dosmagor, no tkmax) ⇒ delta_max was the SOLE structural SGS bug. 1 documented SECONDARY caveat: in weakly-stable layers SAM further shrinks `smix=min(grd, 0.76√tke/N)` (Deardorff, needs a lagged tke; legoESM has only the primary `|S|²−Pr·N²` cutoff, which matches SAM in strongly-stable + unstable layers). |
./docs/dev-notes/CRM_faithful_SAM.md:221:| 183 | **delta_max fix PRELIMINARILY CONFIRMED + buoyancy re-audit.** RCE-cap run (delta_max=1000): burst max\|w\|=10.0 vs the uncapped 8.7; **preliminary comparison (steps 4000-5000): w_RMS=0.49 m/s MID-range vs the uncapped iter-176 0.25 lo-edge** — the delta_max fix cuts the SGS over-mixing ⇒ more resolved convective variance ⇒ w_RMS OFF the lower edge ⇒ BETTER RCEMIP match. CWV unchanged (not SGS-sensitive). **iter-184 FULL-window (4000-7000, apples-to-apples vs the uncapped iter-176): w_RMS 0.25→0.30 (MODEST, still lo-edge — the late-phase weakening dilutes the mean; the BURST window 4000-5500 shows the clearer SGS effect, w_RMS 0.33→0.41); cloud 0.14→0.08 (now lo-edge); 5/7 within spread.** HONEST read: the fix is SAM-FAITHFUL (matches the delta_max source) and raises resolved variance most at PEAK convection; the cloud reduction removes spurious SGS over-spreading (uncapped 0.14 was partly over-mixed detrainment). The earlier "0.41 MID" was the burst-window figure — the equilibrium-ish full-window improvement is real but modest. delta_max justified by source-faithfulness regardless of the small RCEMIP-score shift (5-6/7 either way, within spread). **Buoyancy re-audit (`buoyancy.f90` vs `_moisture_buoyancy_w_half`):** CONFIRMED faithful — `B=g·[ε_v·qv'−q_cond'+(θ'−⟨θ'⟩)/θ₀·(ε_v·q̄v−q̄_cond)]`, ε_v=0.61, **q_cond' = ALL 5 condensate slots (cloud qc/qi + PRECIP qr/qs/qg loading PRESENT)**, deviations-from-horizontal-mean (no spurious mean updraft). Sole gap = the documented uniform-0.5 vs SAM adz-weighted half-level interp on STRETCHED grids (~few %, minor, shared with the dry buoyancy). No new buoyancy bug. |
./docs/dev-notes/CRM_faithful_SAM.md-222-| 185-187 | **Advection audit — found + fixed the momentum-advection fidelity gap behind the lo-edge w_RMS.** SAM active advection (Build:17): SCALARS = MPDATA (monotone ~2nd-order ≈ van_leer ⇒ accepted); MOMENTUM = `advect2_mom_xy/z` (`main.f90:217`) = 2nd-order CENTERED, non-diffusive. legoESM's `--momentum-advection` defaulted to None=van_leer ⇒ the RCE runs used the DIFFUSIVE TVD limiter on u/v/w, damping updraft cores (iter-68 KE-budget: van_leer ≈−0.4‖u‖² dissipative vs centered ≈+0.015). The `centered` option (= SAM `advect2_mom`) is faithful; only the HORIZONTAL momentum was van_leer — the VERTICAL momentum (`_vertical_advection_plane`) is ALREADY centered (= `advect2_mom_z`), vertical scalars use the monotone variant. **θ' vertical advection AUDITED: COMPLETE + centered** — the full `−w·∂θ/∂z` is in the acoustic substep differencing `theta_total` (θ_ref+θ' ⇒ the perturbation self-advection IS included, not a missing term); centered (vs SAM monotone) = the accepted compressible-core diff, conservative (not the cold-point cause). Ran RCE `--momentum-advection centered` (horizontal+vertical both centered, fully SAM-faithful). |
./docs/dev-notes/CRM_faithful_SAM.md-223-| 188 | **Centered-momentum full-window result + a cloud-fraction DIAGNOSTIC bug fixed.** rce_cm (delta_max+centered, FULL 4000-7000): **w_RMS 0.31 (lo-edge)** — the burst window shows the clear SGS+momentum effect (0.33→0.66) but the full-window time-mean DILUTES it (the weak late phase dominates). HONEST: the 2 SAM-faithful fixes reduce numerical over-damping CLEARLY at peak convection, MODESTLY in the time-mean (full-window 0.25→0.31); the residual lo-edge w_RMS is plausibly RESOLUTION/domain-limited FAITHFUL (dx=3 km, 144 km domain, transient — SAM at dx=4 km would sit similarly; the RCEMIP 0.25-0.35 lower end IS this regime), NOT over-damping (that's fixed). Cloud 0.10 full-window (PASS lo-edge) — the earlier 0.05 was a burst-window artifact (anvil develops late). **Found+fixed a cloud-fraction DIAGNOSTIC bug** (`cloud_fraction_profile_plane`): counted ONLY cloud water qc (slot 1), MISSING cloud ICE qi (slot 3 = deep-conv ANVIL) ⇒ under-counted RCE cloud (RCEMIP masks qc+qi). Fixed → qc+qi (robust for warm-only); ice-counting test added. Launched FINAL run (`results/rce_final`: delta_max+centered+ice-cloud) for the definitive comparison; read next iter. |
./docs/dev-notes/CRM_faithful_SAM.md-224-| 189 | **Sponge audit (cold-point θ'-accumulation) — profile EXACT match, but the RCE base is too thin.** SAM `damping.f90`: w-ONLY top damping (line 46 `w/(1+taudamp)`), `taudamp=0.333·tau_max·zzz/(1+zzz)`, `zzz=100·((ν−nub)/(1−nub))²`, base **`nub=0.6`** (params.f90:152). legoESM `sam_rational` (compressible_euler.py:438-440): `zzz=100·frac²`, `sponge_coeff·zzz/(1+zzz)`, frac=(ν−nub)/(1−nub) — **EXACT profile + w-only ✓**. **GAP: the RCE run's base = 0.848, not 0.6** (`--sponge-width` default 5000 m, H=33 km ⇒ ν=1−5000/33000=0.848); matching SAM nub=0.6 needs `sponge_width=0.4·H≈13200 m`. So the RCE sponge is TOO THIN (damps w only above ~28 km vs SAM's ~19.8 km) — a real deviation but LOW impact: the convection (ν~0.2) AND cold-point (ν~0.53) are both BELOW the sponge, so it cannot be the main cold-point-bias cause (that is the w-only sponge leaving TTL θ' undamped — which SAM ALSO does, so the warm cold-point is partly FAITHFUL). Recommend `sponge_width=0.4·H` for RCE; deferred (won't confound the final run). |
./docs/dev-notes/CRM_faithful_SAM.md-225-| 190-192 | **MAJOR RCEMIP realism BUG found + fixed + validated (user: run RCEMIP-small, assess realism, fix bugs): the Wing-IC surface virtual temperature `T_v0`.** Final run (delta_max+centered+ice-cloud) = 6/7 within the RCEMIP spread (CWV 49, T_sfc 297.5, qv 15.4, cloud 0.20) but **cold-point 202.4 K = ~8 K too warm**. DIAGNOSED: θ'≈0 at the cold point ⇒ it IS the reference profile. **ROOT CAUSE: `rcemip_initial_conditions.py` set `T_v0=T_sfc·(1+0.608·q_sfc)`=303 K at SST=300; RCEMIP (Wing 2018 Tab 1) PRESCRIBES a FIXED `T_v0=295 K` for ALL SSTs** (only q_v0 + the surface BC vary). **FIX:** `WING_T_V0=295`; reparameterized the Wing virtual-T/pressure/θ + `make_*` factories by `T_v0` (default 295), DECOUPLED from the SST (driver uses the RCEMIP T_v0; `--T-sfc`=SST is the surface BC only); 9 tests updated, **27 GREEN**. **Cold point now 202.4→194.5 K ✓** (verified actual T(15 km)=194.5). VALIDATED (`rce_fix`): cold-point PASS. BUT the correct (cooler) RCEMIP IC has high CAPE ⇒ a VIOLENT, FAITHFUL spin-up (initial max\|w\|=24, cloud 100%, surface still 293.6 cold +1.9 K/10 h, tropopause overshoot) ⇒ 3/7 at 10 h — NOT equilibrium (the wrong warm-T_v0 coincidentally looked milder/6-7 at 10 h; SAM spins up identically from this prescribed IC). **The T_v0 fix IMPROVES equilibrium faithfulness** (cold point now permanently correct; at equilibrium the surface warms→in-range, cloud→~0.2); a fair RCEMIP comparison needs the ~50-100-day equilibrium (COMPUTE-BOUND). |
./docs/dev-notes/CRM_faithful_SAM.md-226-| 193-194 | **All 4 session bug-fixes confirmed consistent: 71 tests GREEN together** (DNS/LES+delta_max, cloud-ice diagnostic, Wing-IC T_v0, RCEMIP smoke, comparison). **Launched a multi-day equilibrium-trend run** (`results/rce_long`, 50000 steps ≈ 2.9 days, all 4 fixes + the SAM-faithful sponge `--sponge-width 13200` ⇒ base ν=0.6=SAM nub, iter-189) to test the equilibrium projection (surface warming→in-range, cloud organizing→~0.2, tropopause settling) — directly serves "assess realism" since the 10 h run was pure spin-up. Doc shrink (consolidated rows 185-187 + 190-192). **Session scorecard: 4 real bugs fixed via source/realism audit — SGS delta_max, momentum-advection default, cloud-ice diagnostic, Wing-IC T_v0; the convective dynamics AND the IC are now SAM/RCEMIP-faithful; only the equilibrium-statistics comparison remains compute-bound.** |
./docs/dev-notes/CRM_faithful_SAM.md-227-| 195-196 | **Micro sedimentation faithful (cloud-100% = spin-up, NOT a bug) + RCE momentum default → SAM-faithful.** (195) `morrison.py` sedimentation IS computed + applied for ALL species (`sed_r/i/s/g`:749-768 → added to the mass tendencies 823-837; DCS=250 µm = SAM `clice_snow_Dauto`); the cloud-100% / anvil q_cloud 0.57 g/kg with q_precip≈0 = the violent spin-up detrains ice faster than it slow-sediments over the small 144 km domain (rain/snow/graupel fall fast ⇒ low in-column precip). NOT a 5th bug. (196) `run_rcemip_plane.py` `--momentum-advection` default None=van_leer → **`centered`** (= gSAM `advect2_mom`) ⇒ "run RCEMIP small" is momentum-faithful BY DEFAULT (smokes build config directly ⇒ unaffected). |
./docs/dev-notes/CRM_faithful_SAM.md-228-| 197-199 | **Long-run cold-drift + cloud-100% + near-uniform field DIAGNOSED PHYSICAL (compute-bound), not bugs; surface-flux wiring VERIFIED faithful.** The cold-point cooling below range (190→182) is the convective OVERSHOOT + moist-adiabatic adjustment of the cold (RCEMIP-correct, high-CAPE) IC: upper-trop θ' warms toward the moist adiabat (+50 K, DECELERATING → plateau ⇒ slowly EQUILIBRATING, not diverging), lower-strat overshoot-cools; ~days to equilibrium (COMPUTE-BOUND). Surface flux faithful (`_make_surface_flux_physics`→`compute_sam_oceflx_fluxes`→ lowest-level T+q tendency :197/205); stuck-cold surface (294 vs SST 300) = flux-vs-convective-cooling balance. Open: the small-domain run is near-uniform (horizontal std ~1 K, cloud 100%, max\|w\| dying 24→1.5, BL seed damps 0.5→0.1) — does it re-organize into cells? (the warmer-IC `rce_final` DID organize to cloud 0.20 ⇒ legoESM CAN; CAPE/IC-dependent). No gSAM cross-check (unbuildable). |
./docs/dev-notes/CRM_faithful_SAM.md-229-| 200,202 | **Vertical θ' advection was CENTERED (non-monotone) in the acoustic substep — a real FAITHFULNESS gap (SAM advects ALL scalars monotonically; tracers + horizontal θ' were already van_leer). FIXED with a monotone van-Leer TVD option** (`_theta_vert_advection_van_leer_kernel`, reuses the shared `van_leer_face_values`; both acoustic kernels, gated by config `acoustic_theta_advection` / driver `--acoustic-theta-advection`, default `"centered"` = BYTE-IDENTICAL, decoupled from the implicit w-solve). Tests: helper bit-identical to the tested plane van_leer + TVD (no new extrema vs centred OVERSHOOT) + full SI dycore finite; **93 tests pass (default byte-identical: 51 dycore-unit + 8 NH benchmarks + 34 plane/smoke).** **iter-200 hypothesis that this caused the cold drift → FALSIFIED by the A/B (202):** `rce_vl`(van_leer) vs `rce_long`(centred), same IC: van_leer cools the cold-point FASTER (cp 182.8 vs 188.4 @ step 5000) because LESS diffusive ⇒ preserves updraft cores (θ' −39/+64 vs −17/+36) ⇒ STRONGER overshoot. ⇒ the cold drift is PHYSICAL (confirms 197-198), not numerics. van_leer STANDS as a scheme-faithfulness improvement (opt-in; default stays centred — it doesn't "fix" the drift). Honesty: iter-200 over-attributed a physical effect to numerics. |
./docs/dev-notes/CRM_faithful_SAM.md-230-| 203 | **RCEMIP radiation setup AUDITED — faithful to gSAM's actual config (no gap).** `run_rcemip_plane.py` RAD-2/3/4: trace gases CO2/CH4/N2O = **355/1700/320** ppmv/ppbv — the comment notes these were VERIFIED from gSAM's `RUNDATA/rrtmg_lw.nc` MLS VMRs (what gSAM ACTUALLY runs with nxco2=1, NOT the Wing-2018 protocol 348; the oracle is gSAM) — plus SAM Briegleb zenith-dependent ocean albedo + 0.07 diffuse (RCEMIP), MLS ozone, RCEMIP insolation **409.6 W/m²** (S0=551.58, zenith 42.05°). ⇒ the radiative-convective forcing that SETS the RCE equilibrium is faithful (modulo rrtmgp vs RRTMG, the accepted scheme diff). **Codex review: environmentally IMPRACTICAL (3 attempts) — codex shell is bwrap-blocked for file reads, and a pasted-code review auto-backgrounds + gets SIGTERM'd (exit 143) at the Ralph turn boundary. van_leer-code correctness rests on the bit-identical-to-tested-routine + TVD unit tests + 93-test byte-identical default + self-review.** |
./docs/dev-notes/CRM_faithful_SAM.md:231:| 205-206 | **BUG #6 (cloud-ice fall exponent) + micro process-rate audit.** **#6:** `fall_b_i` 1.0 (M2005-orig) → **0.865** (gSAM `clice_fall_b` MK tune) — the 1.0 fell ~3× too slow (v=0.01/0.07/0.7 vs gSAM 0.03/0.24/1.78 m/s) ⇒ anvil ice over-accumulated; auto-propagates through cons27/cons28/V_t_i/V_n_i. Audit CONFIRMS faithful (no other bug): rain (AR=841.99667/BR=0.8), snow (AS=11.72/BS=0.41), graupel (AG=19.3/BG=0.37); scheme flags `doicemicro=T, dograupel=T, dohail=F` (⇒ graupel ρ_g=400 not hail), `dosb_warm_rain=F` (⇒ KK2000); KK2000 warm rain EXACT — `PRC=1350·qc^2.47·Nc[#/cm³]^−1.79` + `PRA=67·(qc·qr)^1.15`; DCS=250µm. Guard test; 11 fall + 331 micro/dycore tests pass (10 `-k`-caught fails pre-existing/unrelated). |
./docs/dev-notes/CRM_faithful_SAM.md-232-| 207-208 | **Cloud-100% reframed (small-domain anvil, NOT a threshold artifact) + bug #6 validation A/B launched.** The compare metric is the PEAK domain cloud fraction (`np.max(cloud)` over levels; RCEMIP 0.08-0.30). legoESM's peak ≈1.0 = the ANVIL covering the WHOLE non-organized 144 km domain — a REAL over-cloud vs RCEMIP, but SMALL-DOMAIN/compute-bound (organized RCE confines the anvil to ~20-30%). **Cloud-threshold audit:** gSAM `statistics.f90:674` uses `coef=min(1e-5, 0.01·qsatw(T,p))` (1e-5 warm trop → ~1e-7 cold anvil); legoESM uses fixed 1e-6 (RCEMIP-protocol, Wing 2018 Tab A2). gSAM unbuildable ⇒ we compare to RCEMIP-PUBLISHED (1e-6 convention) ⇒ 1e-6 is correct, NOT a bug; the cloud-100% peak is anvil-condensate-thick (0.14-0.57 g/kg ≫ both thresholds), robust to either. **Launched `rce_icefall`** = rce_long's CLI with ONLY bug #6's `fall_b_i=0.865` differing — tests whether faster ice fall thins the anvil ⇒ lower peak cloud (A/B vs rce_long b=1.0). Found rce_long's iter-200 kill had FAILED (ran to step 28800; anvil ice declined 0.57→0.14 g/kg as the spin-up settled, cloud stayed 1.0). A/B reads next iter. |
./docs/dev-notes/CRM_faithful_SAM.md-233-| 209-212 | **BUG #7 (ice fall density) + fall-speed/micro audit COMPLETE + A/B + snow 2-mom.** **#7:** ice fall DENSITY exponent 0.54→**0.35** (gSAM `AIN=(RHOSU/RHO)^0.35·AI` Ikawa-Saito, used in `UMI=AIN·CONS28/LAMI^BI`; rain/snow/graupel correctly KEEP 0.54) — was ~25% too fast in thin air; `morrison.py` `dum_i=(ρ_su/ρ)^0.35` for V_t_i+V_n_i. **#6+#7 ⇒ ice fall matches gSAM EXACTLY.** Verified faithful (no more bugs): ALL caps (ice `1.2·(ρsu/ρ)^0.35`, rain `9.1·dum`, snow/graupel `1.2·dum`/`20·dum`), ice deposition (`EPSI=2π·N_i·ρ·DV/LAMI`≡gSAM, `PRD=EPSI·(qv−qsi)/ABI`), LAMI slope (`CONS12=π·ρ_ci`, ρ_ci=500), 2-MOMENT snow (N_s slot 9 ⇒ PSD fall + PRDS/PSACWS/PSMLT/NSAGG — the old "single-moment snow gap" is stale). **rce_icefall A/B (b=0.865 vs rce_long b=1.0):** anvil q_cloud 0.53-0.67 g/kg in BOTH ⇒ the ~3× faster ice fall did NOT cut the cloud-100% — the anvil is DETRAINMENT-dominated + small-domain (fixes are source-faithful, not the cloud-100% cause). 82 micro/dycore tests GREEN ⇒ M2005 microphysics THOROUGHLY faithful. |
./docs/dev-notes/CRM_faithful_SAM.md-234-| 213 | **Closed the documented secondary SGS gap (#8): SAM dosmagor's WEAKLY-stable Deardorff mixing-length limit.** Verified SAM `tke_full.f90:285-298` dosmagor DOES apply, where N²>0: `smix=min(grd, max(0.1·grd, √(0.76·tk/(Ck·√N²))))` + `Cee=Ce1+Ce2·(smix/grd)` ⇒ `tk=√(Ck³/Cee·(|S|²−Pr·N²))·smix²` (Ck=0.1, Ce=Ck³/Cs⁴, Ce1/Ce2=Ce/0.7·{0.19,0.51}). legoESM used the pure `(Cs·Δ)²·|S|` (smix=grd always) ⇒ OVER-mixed in WEAKLY-stable layers (the |S|²−Pr·N² cutoff already matched SAM in unstable + strongly-stable). **FIX:** `smagorinsky_stability_length` (config + `--smag-stability-length`), in BOTH serial + halo K_m (2-pass: smix=grd tk estimate → Deardorff smix, as SAM lags tk; reduces EXACTLY to the pure form where N²≤0). Default False = BYTE-IDENTICAL. Tested (unstable identical, weakly-stable strictly reduces K_m, serial mirrors halo); 8 closure + 24 dycore GREEN. Minor effect (stratosphere/inversion, NOT the convective signal) but closes the LAST characterized SGS gap ⇒ the SGS now FULLY matches SAM dosmagor (with `--smag-stability-length --no-smag-wall-damping`). |
./docs/dev-notes/CRM_faithful_SAM.md-235-| 215 | **Cloud effective-radius audit — ICE r_eff (dominant anvil) FAITHFUL; liquid 14µm = SAM ocean value.** SAM-M2005 uses the M2005 PSD-COMPUTED r_eff in radiation (`douse_reffc/reffi=.true.` default), not the CAM T-fallback. **ICE: SAM `EFFI=3/LAMI/2·1e6=1.5/LAMI` (mp_graupel:4866) = legoESM `EFFI=1.5/LAMI` (radiation/integration.py:914) EXACTLY** (same faithful LAMI slope, iter-211) ⇒ the anvil-ice r_eff = the DOMINANT RCE cloud radiative forcing = faithful. **LIQUID: legoESM 14µm fixed = SAM `computeRe_Liquid` OCEAN value (rliqocean=14, cam_rad_parameterizations.f90); SAM-M2005's PSD `reffc=(pgam+3)/lambda/2` (~8-14µm) varies** ⇒ a minor approximation (warm LOW clouds, SECONDARY to the anvil; 14µm is within the PSD range). + gases/albedo/ozone/insolation already faithful (iter-203) ⇒ the cloud radiative forcing is faithful in its dominant (ice) component; the fixed-14µm liquid is a minor documented simplification. |
./docs/dev-notes/CRM_faithful_SAM.md-236-| 216 | **User-requested SAM-tail refinements: liquid r_eff PSD + M1b deposition split DONE; graupel self-collection NOT added (gSAM has none).** (1) **Liquid r_eff PSD (#9):** `compute_cloud_properties` now computes `reffc=(PGAM+3)/(2·LAMC)` from q_c+N_c (PGAM Martin-1994 from N_c[#/cm³], LAMC from q_c/N_c per-mass; mp_graupel:495/1676-1692) when N_c (slot 6) present, replacing the fixed 14µm (=SAM CAM-ocean fallback); SAM-M2005 uses this PSD reffc by default (`douse_reffc=T`). Threaded `n_cloud` through the radiation; bounded by SAM's 1-60µm LAMMIN/LAMMAX. Test (known case ~12µm + q_c/N_c monotonicity + no-N_c/liquid-free fallback). (2) **M1b deposition size-split (#10):** the cloud-ice depositional growth PAST DCS (tail `1−DUM`, `DUM=1−exp(−LAMI·DCS)(1+LAMI·DCS)`) now routes to SNOW when present (was ALL→cloud ice; mp_graupel:3466-3481), via `prds` ⇒ conservation-EXACT (dq_i_dep reduced by the same amount; vapour/latent/dq_s consistent; prds≥0 when dep>0 same-sign). 101 micro tests pass (1 fail pre-existing None-field). (3) **Graupel self-collection — NOT added: gSAM has NONE** (NG3DTEN = melt+conversion+sublimation+sedimentation only; self-collection is RAIN/NRAGG + SNOW/NSAGG ONLY). legoESM lacking it is FAITHFUL; adding would be UNFAITHFUL. legoESM's snow NSAGG (which SAM DOES have) already present (`snow_self_aggregation_nsagg`, dN_s_dt:948-952). Broad micro+radiation regression: 146 pass, reffc+M1b SAFE (column-local). **Fixed 2 PRE-EXISTING broken tests (not from this session):** (a) `test_finite_outputs` iterated over optional None fields (2-mom number tendencies) → None-skip ⇒ 68 micro GREEN (6 schemes' finite-checks restored); (b) `test_sum_plane_tendencies` used the obsolete `_make_surface_flux_physics(Cd,Ch,q_sfc)` → current `(T_sfc,p_sfc)` ⇒ 17 GREEN. **GATE diurnal-SW finding:** gSAM `CASES/GATE_IDEAL/prm` uses the DIURNAL cycle (latitude0=8.5, day0=244.6, no doperpetual); legoESM GATE uses daily-mean (RAD-8). A minor gap, SECONDARY to the GATE dx=1000m resolution limit (GATE doesn't self-trigger regardless). |
--
./packages/tools/legoesm/forcing/jra55_do.py-137-})
./packages/tools/legoesm/forcing/jra55_do.py-138-
./packages/tools/legoesm/forcing/jra55_do.py-139-#: Plausibility ranges per variable (units as in the schema above).
./packages/tools/legoesm/forcing/jra55_do.py-140-#: Values outside these are logged as warnings at cache-build time —
./packages/tools/legoesm/forcing/jra55_do.py-141-#: a guard against silently bad source data, not a hard rejection.
./packages/tools/legoesm/forcing/jra55_do.py:142:_PLAUSIBLE_RANGE: dict[str, tuple[float, float]] = {
./packages/tools/legoesm/forcing/jra55_do.py-143-    "uas":    (-100.0, 100.0),
./packages/tools/legoesm/forcing/jra55_do.py-144-    "vas":    (-100.0, 100.0),
./packages/tools/legoesm/forcing/jra55_do.py-145-    "tas":    (180.0, 340.0),     # K
./packages/tools/legoesm/forcing/jra55_do.py-146-    "huss":   (0.0, 0.06),        # kg/kg
./packages/tools/legoesm/forcing/jra55_do.py-147-    "psl":    (87000.0, 110000.0),  # Pa
--
./packages/tools/legoesm/forcing/jra55_do.py-706-    return flat.reshape(arr.shape)
./packages/tools/legoesm/forcing/jra55_do.py-707-
./packages/tools/legoesm/forcing/jra55_do.py-708-
./packages/tools/legoesm/forcing/jra55_do.py-709-def _check_plausible(var: str, arr: np.ndarray) -> None:
./packages/tools/legoesm/forcing/jra55_do.py-710-    """Warn if values lie outside the documented plausible range."""
./packages/tools/legoesm/forcing/jra55_do.py:711:    lo, hi = _PLAUSIBLE_RANGE[var]
./packages/tools/legoesm/forcing/jra55_do.py-712-    a_min = float(np.nanmin(arr))
./packages/tools/legoesm/forcing/jra55_do.py-713-    a_max = float(np.nanmax(arr))
./packages/tools/legoesm/forcing/jra55_do.py-714-    if a_min < lo or a_max > hi:
./packages/tools/legoesm/forcing/jra55_do.py-715-        print(
./packages/tools/legoesm/forcing/jra55_do.py-716-            f"[jra55_do] WARNING: {var} range [{a_min:.3g}, {a_max:.3g}] "
--
./tests/parallel/test_atm_latlon_2d_tiling.py-221-
./tests/parallel/test_atm_latlon_2d_tiling.py-222-@pytest.mark.parametrize("p_lat,p_lon", [(1, 4), (2, 2)])
./tests/parallel/test_atm_latlon_2d_tiling.py-223-@pytest.mark.parametrize("halo", [1, 2])
./tests/parallel/test_atm_latlon_2d_tiling.py-224-def test_partner_fold_emits_no_all_gather(p_lat, p_lon, halo):
./tests/parallel/test_atm_latlon_2d_tiling.py-225-    """EVEN p_lon pads must lower with ZERO all-gather (the pole fold is
./tests/parallel/test_atm_latlon_2d_tiling.py:226:    the antipodal partner ppermute — collective-permute in HLO) while
./tests/parallel/test_atm_latlon_2d_tiling.py-227-    still matching the serial pad window BIT-exactly (belt+braces with
./tests/parallel/test_atm_latlon_2d_tiling.py-228-    gate 1: this is the direct exercise of partner_pole_fold_window
./tests/parallel/test_atm_latlon_2d_tiling.py-229-    through its only production call site)."""
./tests/parallel/test_atm_latlon_2d_tiling.py-230-    from legoesm.grids.halo_latlon import pad_halo_latlon_3d_local
./tests/parallel/test_atm_latlon_2d_tiling.py-231-
--
./tests/parallel/test_atm_latlon_2d_tiling.py-234-    field = jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)))
./tests/parallel/test_atm_latlon_2d_tiling.py-235-    fn, sharded, hlo = _lowered_pad(mesh, field, halo)
./tests/parallel/test_atm_latlon_2d_tiling.py-236-    assert "all-gather" not in hlo, (
./tests/parallel/test_atm_latlon_2d_tiling.py-237-        "even-p_lon pole fold must not emit all-gather "
./tests/parallel/test_atm_latlon_2d_tiling.py-238-        "(partner ppermute lever)")
./tests/parallel/test_atm_latlon_2d_tiling.py:239:    assert "collective-permute" in hlo
./tests/parallel/test_atm_latlon_2d_tiling.py-240-    out = np.asarray(jax.device_put(fn(sharded), NamedSharding(mesh, P())))
./tests/parallel/test_atm_latlon_2d_tiling.py-241-    serial = np.asarray(pad_halo_latlon_3d_local(field, halo))
./tests/parallel/test_atm_latlon_2d_tiling.py-242-    nl, w = N_LAT // p_lat, N_LON // p_lon
./tests/parallel/test_atm_latlon_2d_tiling.py-243-    hl, hw = nl + 2 * halo, w + 2 * halo
./tests/parallel/test_atm_latlon_2d_tiling.py-244-    for r in range(p_lat):
--
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-6-split-explicit re-association floor.  A staggered off-by-one in the wide
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-7-v-exchange, a reach under-budget at a band cut, or a pole-flag error in the
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-8-extended geometry shows up here as an O(1e-3+) band-cut mismatch.
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-9-
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-10-Also gates the POINT of the wide path mechanically: the compiled sharded
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py:11:step's WHILE-BODY ``collective-permute`` count must DROP vs the standard
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-12-config — per-iteration subcycle communication is exactly what the wide
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-13-exchange removes (static op totals are NOT runtime message counts: loop-body
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-14-ops run n_substeps times, the wide path's unrolled exchanges run once).
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-15-
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-16-Run: ``XLA_FLAGS=--xla_force_host_platform_device_count=4 \
--
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-137-    path's unrolled chunk exchanges each run once.  The structural claim to
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-138-    gate is therefore: the wide path leaves NO barotropic halo collective
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-139-    inside a loop body — its substeps are communication-free.
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-140-
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-141-    Parses the HLO text into computation blocks (``%name (args) -> ... {``
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py:142:    to the closing ``}``), counts collective-permutes per block, and sums
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-143-    the counts over blocks referenced as ``body=%name`` by while ops.
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-144-    """
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-145-    import re
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-146-
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-147-    blocks: dict[str, int] = {}
--
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-154-            continue
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-155-        if name is not None:
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-156-            if line.startswith("}"):
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-157-                blocks[name] = count
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-158-                name = None
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py:159:            elif "collective-permute" in line and "done" not in line:
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-160-                count += 1
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-161-    body_names = set(re.findall(r"body=%?([\w\.\-]+)", hlo_text))
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-162-    return sum(blocks.get(b, 0) for b in body_names)
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-163-
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-164-
--
./docs/dev-notes/fv3_faithful.md-52-  (prevents silent 180° re-break); latlon branch confirmed self-consistent (native data+label; staggered
./docs/dev-notes/fv3_faithful.md-53-  fields legitimately differ in size, so no guard). barotropic_wave 3/3 PASS + 17/17 regrid tests post-guard.
./docs/dev-notes/fv3_faithful.md-54-  Pending: resolution-convergence check.
./docs/dev-notes/fv3_faithful.md-55-
./docs/dev-notes/fv3_faithful.md-56-## ❗ THE ONE REMAINING GAP — experimental SW FB-port edge instability (production SW is STABLE + faithful-in-results)
./docs/dev-notes/fv3_faithful.md:57:CONFIRMED FV3 MISMATCH (codex iter109): production SW (operators_cdgrid.py:1167) + 3D PE (:445) use CENTERED
./docs/dev-notes/fv3_faithful.md-58-`zeta_corner*v_d`; FV3 upwind donor-cell. The staggered c_sw→d_sw port (`fv3_fb_sw_step`, upwind
./docs/dev-notes/fv3_faithful.md-59-`_vorticity_flux`) is EXPERIMENTAL + has a weak edge instability (W2 C36 day2 NaN). Production's co-located
./docs/dev-notes/fv3_faithful.md-60-D→A-avg scheme suppresses it. FULLY CHARACTERIZED (iter123-134, two codex reviews drove it to ground truth):
./docs/dev-notes/fv3_faithful.md-61-- **TYPE (iter128 genuine Arnoldi, dim 23760):** weakly-UNSTABLE spectrum **ρ(M')≈1.0019>1** (real eigenvalue
./docs/dev-notes/fv3_faithful.md-62-  cluster) + strong NON-NORMAL TRANSIENT growth (K=50 optimal ~1.007/step ≫ the eigenvalue). The earlier
--
./docs/dev-notes/fv3_faithful.md-139-
./docs/dev-notes/fv3_faithful.md-140-## NEW LOOP (2026-06-04) — re-verification on the federated `packages/` tree
./docs/dev-notes/fv3_faithful.md-141-Source-of-truth moved to `packages/core/legoesm/` (uv-workspace federation, #361); run via `uv run pytest`.
./docs/dev-notes/fv3_faithful.md-142-Baseline: 36/36 corner+edge fidelity tests pass (d_sw5 corner div/corr, divergence_corner, dgrid_corner_fill).
./docs/dev-notes/fv3_faithful.md-143-
./docs/dev-notes/fv3_faithful.md:144:- **iter1 — REFUTED the iter147 c2l-halo edge-bug hypothesis (by measurement).** iter147 had claimed the
./docs/dev-notes/fv3_faithful.md-145-  covariant→geographic step in the cross-face D-grid halo (`ext_vector_dgrid` step 1, `duogrid.py`) was the
./docs/dev-notes/fv3_faithful.md-146-  edge-artifact source and "fixed" it with a non-orthogonal cos_angle/sin_angle/cosa_s formula. MEASURED the
./docs/dev-notes/fv3_faithful.md-147-  hand-rolled transform MATRIX vs FV3's exact `c2l_ord2` z-matrix (a11..a22, built from the previously
./docs/dev-notes/fv3_faithful.md-148-  UNUSED-but-faithful `init_cubed_to_latlon` port as oracle): deviation is **2nd-order convergent and
./docs/dev-notes/fv3_faithful.md-149-  sub-1e-3 by C48 in the only cells the halo consumes** (rings ≤2 from a face edge): d=0 ring max
--
./docs/dev-notes/planning/eady_eddy_resolving_ralph.md-414-  exits 144. RELIABLE way = run the python directly as a `run_in_background=true` Bash task
./docs/dev-notes/planning/eady_eddy_resolving_ralph.md-415-  (harness-tracked, persists, notifies). Don't nohup-detach.
./docs/dev-notes/planning/eady_eddy_resolving_ralph.md-416-**If B_h=2.5e11 is clean (spike gone AND BCI grows ~σ_Eady):** WENO5-track answer. Then
./docs/dev-notes/planning/eady_eddy_resolving_ralph.md-417-sweep B_h to the minimum that works, re-test --no-sponge, push ~10 km, apply to NEMO, commit.
./docs/dev-notes/planning/eady_eddy_resolving_ralph.md-418-
./docs/dev-notes/planning/eady_eddy_resolving_ralph.md:419:### Iteration 7 (FIX CONFIRMED)
./docs/dev-notes/planning/eady_eddy_resolving_ralph.md-420-**A_h=5e3 (Veros Laplacian viscosity, C_smag=0) KILLS the grid-scale spike.** weak-120
./docs/dev-notes/planning/eady_eddy_resolving_ralph.md-421-dt=300, day 5: max|u|=**0.93 m/s** vs 2.6–4.5 in every no-A_h run. EKE clean (1.96e-10).
./docs/dev-notes/planning/eady_eddy_resolving_ralph.md-422-→ The WENO5-track fix is **harmonic Laplacian viscosity** (the piece my rebuild dropped;
./docs/dev-notes/planning/eady_eddy_resolving_ralph.md-423-biharmonic Smag alone is too scale-selective). Awaiting full 40-day growth+saturation.
./docs/dev-notes/planning/eady_eddy_resolving_ralph.md-424-**Next:** confirm full run clean (λ ratio ~1, physical saturation); then re-test `--no-sponge`
--
scripts/matrix/run_atmosphere_test_matrix.py-904-this NaNs at day 22.5 — STILL INSUFFICIENT for 30-day."""
scripts/matrix/run_atmosphere_test_matrix.py-905-
scripts/matrix/run_atmosphere_test_matrix.py-906-_CFL_SAFETY_VERY_LONG_TIME: float = 0.154
scripts/matrix/run_atmosphere_test_matrix.py-907-"""iter 80 calibration: dt=50 at C96.  iter-79 found dt=100 NaN's
scripts/matrix/run_atmosphere_test_matrix.py-908-at day 22.5; this halves dt further as the next attempt at 30-day
scripts/matrix/run_atmosphere_test_matrix.py:909:stability.  iter 99 EMPIRICALLY CONFIRMED full 30-day finite at
scripts/matrix/run_atmosphere_test_matrix.py-910-C96 (max|u|=20.14 m/s, max|v|=11.84 m/s, 51840 steps, 1755 s
scripts/matrix/run_atmosphere_test_matrix.py-911-wall).  iter-85 linear-in-1/dt prediction held: dt=50 was
scripts/matrix/run_atmosphere_test_matrix.py-912-predicted to NaN at day 45; never reached because the 30-day run
scripts/matrix/run_atmosphere_test_matrix.py-913-completed finite at day 30."""
scripts/matrix/run_atmosphere_test_matrix.py-914-
--
scripts/matrix/run_atmosphere_test_matrix.py-6066-            from tests.test_cases.dcmip2025 import dcmip25_tc3_init
scripts/matrix/run_atmosphere_test_matrix.py-6067-            state, hcoord, tmetric, small_grid = dcmip25_tc3_init(
scripts/matrix/run_atmosphere_test_matrix.py-6068-                grid, n_levels=nlev)
scripts/matrix/run_atmosphere_test_matrix.py-6069-            grid = small_grid
scripts/matrix/run_atmosphere_test_matrix.py-6070-            #
scripts/matrix/run_atmosphere_test_matrix.py:6071:            # ⚠️ new_test_dycores iter-108 CAUTION (CONFIRMED by iter-123):
scripts/matrix/run_atmosphere_test_matrix.py-6072-            # TC3 cube BLOWS UP at full mode at step 2250 (day 0.01,
scripts/matrix/run_atmosphere_test_matrix.py-6073-            # 8.3 min sim time) with `metric 1271.0 > threshold 1000.0`
scripts/matrix/run_atmosphere_test_matrix.py-6074-            # (max|w| exceeds threshold).  Same iter-12..17 NH bundle
scripts/matrix/run_atmosphere_test_matrix.py-6075-            # that gives PASS at quick mode (|w|=7.36 m/s per iter-7
scripts/matrix/run_atmosphere_test_matrix.py-6076-            # claim) is INSUFFICIENT for full-duration cube stability.
--
scripts/matrix/run_atmosphere_test_matrix.py-8069-                                                 C72, dt=100 at C96;
scripts/matrix/run_atmosphere_test_matrix.py-8070-                                                 NaN day 22.5 at C96
scripts/matrix/run_atmosphere_test_matrix.py-8071-                                                 30d per iter 79)
scripts/matrix/run_atmosphere_test_matrix.py-8072-                                  very_long_time iter-80 (dt=67 at
scripts/matrix/run_atmosphere_test_matrix.py-8073-                                                 C72, dt=50 at C96;
scripts/matrix/run_atmosphere_test_matrix.py:8074:                                                 iter-99 CONFIRMED
scripts/matrix/run_atmosphere_test_matrix.py-8075-                                                 30d finite at C96
scripts/matrix/run_atmosphere_test_matrix.py-8076-                                                 max|u|=20.14)
scripts/matrix/run_atmosphere_test_matrix.py-8077-                                  auto           RECOMMENDED: short_time
scripts/matrix/run_atmosphere_test_matrix.py-8078-                                                 at n<96, very_long_time
scripts/matrix/run_atmosphere_test_matrix.py-8079-                                                 at n>=96.  Validated
--
scripts/cluster/scaling_derecho/README.md-483-|---|---|
scripts/cluster/scaling_derecho/README.md-484-| `ocean_gpu_scaling.pbs` | OCEAN weak+strong on one GPU node via `scripts/bench/bench_ocean_latlon_spmd_scaling.py` (full lat-lon C-grid step sharded over 1/2/4 A100; fail-fast `--parity-gate` + `--check-conservation` smoke first). Plain GPU env — no mpi4jax. |
scripts/cluster/scaling_derecho/README.md-485-| `ocean_cpu_scaling.pbs` | OCEAN weak+strong CPU-MPI rank ladder (`bench_ocean_mpi_scaling.py`, `legoesm-mpi` env), with a 2-rank parity+conservation smoke. |
scripts/cluster/scaling_derecho/README.md-486-| `gpu_multinode_scaling.pbs` | MULTI-NODE GPU lanes over jax.distributed + NCCL: A = cube `--cs-spmd` (6 GPU / 2 nodes), C = atm lat-lon `--multicontroller` (8 GPU), D = ocean `--multicontroller` (8 GPU); plus the optional route-A CUDA-aware mpi4jax lane (`RUN_ROUTEA=1`, needs the overlay env) and the comm-tuning A/B ladder (`RUN_TUNE=1`, lane T below). |
scripts/cluster/scaling_derecho/README.md-487-| `build_nccl_ofi.sh` | Login-node build of **aws-ofi-nccl** against Derecho's Cray libfabric (no NCCL build dep — the plugin vendors the net-API headers and is dlopen'd by the jax-wheel NCCL). |
scripts/cluster/scaling_derecho/README.md:488:| `diagnosis.pbs` | BOTTLENECK diagnosis via `scripts/bench/run_scaling_diagnosis.py` — per-phase halo bandwidth, reduction latency, roofline, compute/comm overlap, and the **static collective census** (collective-permute + all-reduce + all-gather per step) that the throughput jobs above do NOT capture. Climbs the cube face-shard `1 2 3` ladder (must divide 6) so the message-count-vs-shard curve — the LATENCY-bound anti-scaling signal — is recorded. `qsub -v MODE=census` for counts only. |
scripts/cluster/scaling_derecho/README.md-489-
scripts/cluster/scaling_derecho/README.md-490-NCCL on Slingshot-11 has NO native CXI support: without the plugin the
scripts/cluster/scaling_derecho/README.md-491-multi-node lanes fall back to TCP sockets over `hsn` (correct, 2-3x slower
scripts/cluster/scaling_derecho/README.md-492-comm — loud warning, fine for shakeout). For production numbers:
scripts/cluster/scaling_derecho/README.md-493-
--
scripts/cluster/scaling_derecho/README.md-520-
scripts/cluster/scaling_derecho/README.md-521-## 2026-07 lane T: comm-tuning A/B ladder (`RUN_TUNE=1`)
scripts/cluster/scaling_derecho/README.md-522-
scripts/cluster/scaling_derecho/README.md-523-Once the route-B lanes are green on this machine, the remaining strong-
scripts/cluster/scaling_derecho/README.md-524-scaling headroom at small tiles is **per-step message count × latency**
scripts/cluster/scaling_derecho/README.md:525:(census: cube 46 collective-permutes/step at 6 devices with field packing
scripts/cluster/scaling_derecho/README.md-526-already at floor; atm latlon 41 → 29 behind the fused-halo flag — see
scripts/cluster/scaling_derecho/README.md-527-`docs/performance/scaling/spmd_message_census_2026-07-08.md`). Lane T runs
scripts/cluster/scaling_derecho/README.md-528-the ranked rungs as same-allocation A/B arms (a fresh `base` control arm is
scripts/cluster/scaling_derecho/README.md-529-re-run in the same job — never compare against an earlier job's numbers):
scripts/cluster/scaling_derecho/README.md-530-
--
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-66-
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-67-### Iteration 5 — stabilization SOLVED for WENO schemes; SM2/QG2 + matrix are GPU-gated — 2026-06-15
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-68-- **CORE OBJECTIVE SOLVED for the no-closure schemes.** Bare vector_invariant (no closure, no WENO)
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-69-  also blows (48×32×50 day 34) — consistent: legoESM's eddy-resolving momentum under-dissipates,
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-70-  period. The `--stabilize` backstop (A_h=1000+C_smag=0.1) fixes the WENO/upwind schemes (W9V/W9D/UP3),
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md:71:  CONFIRMED stable at 1/8° (160×128). **These 3 schemes are matrix-ready.**
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-72-- **SM2/QG2 follow-up (smaller):** their explicit closures (OM4p25/QG-Leith at nominal coeffs) are
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-73-  insufficient at 1/8° (matrix: SM2 d24, QG2 d15). The `--stabilize` backstop can't apply (double-
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-74-  friction guard). To matrix them: either (a) add a `lateral_friction_allow_backstop` flag that
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-75-  relaxes the guard so a small A_h rides alongside the closure (a numerics change → review), or
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-76-  (b) stronger OM4p25/QG-Leith coefficients, or (c) run them at 1/16° (less dissipation needed). Can't
--
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-80-  `run_silvestri_baroclinic_jet.py --scheme {W9V,W9D,UP3} --resolution 160x128 --days 1000 --stabilize`.
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-81-- **STATUS: the §5 stabilization (the hard diagnostic+fix) is DONE.** What remains — running the matrix
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-82-  + SM2/QG2 tuning — is GPU compute, impossible while both GPUs are externally occupied. Pausing the
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-83-  loop (like the parent reproduction loop) until GPU is free; everything is committed on PR #475.
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-84-
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md:85:### Iteration 4 — D2 CONFIRMED at 160×128 (1/8°); --stabilize wired — 2026-06-15
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-86-- **160×128×50 + A_h=1000 + C_smag=0.1 (weno9): STABLE to 18 days** (max|u| flat ~0.06 through
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-87-  days 5/10/15/18), where un-damped weno9 BLEW at day 11. The backstop decisively fixes §5 at the
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-88-  paper's coarsest production resolution. (max|u|~0.06 = eddies still developing slowly; 1/8°
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-89-  under-resolves L_d so growth is slow, but it's STABLE — the point.)
./docs/dev-notes/planning/silvestri_jet_stabilization_ralph.md-90-- **Wired `--stabilize`** into `build_silvestri_baroclinic_jet_setup(stabilize=True)` + the driver:
--
scripts/cluster/scaling_derecho/diagnosis.pbs-10-# ===========================================================================
scripts/cluster/scaling_derecho/diagnosis.pbs-11-# BOTTLENECK DIAGNOSIS on a Derecho GPU node (4x A100-40GB, NVLink).
scripts/cluster/scaling_derecho/diagnosis.pbs-12-#
scripts/cluster/scaling_derecho/diagnosis.pbs-13-# Runs scripts/bench/run_scaling_diagnosis.py — the per-phase bottleneck tool
scripts/cluster/scaling_derecho/diagnosis.pbs-14-# (halo bandwidth, reduction latency, roofline, compute/comm overlap, and the
scripts/cluster/scaling_derecho/diagnosis.pbs:15:# STATIC collective census: collective-permute + all-reduce + all-gather per
scripts/cluster/scaling_derecho/diagnosis.pbs-16-# step) — which the throughput jobs (scaling_gpu.sh, ocean_gpu_scaling.pbs)
scripts/cluster/scaling_derecho/diagnosis.pbs-17-# do NOT capture.  This is the lane that answers "WHERE is the strong-scaling
scripts/cluster/scaling_derecho/diagnosis.pbs-18-# loss", not just "what is the SYPD".
scripts/cluster/scaling_derecho/diagnosis.pbs-19-#
scripts/cluster/scaling_derecho/diagnosis.pbs-20-# Why this matters (measured campaign readings, derecho_levante_sota_review):
scripts/cluster/scaling_derecho/diagnosis.pbs-21-#   * f64 and f32 GPU strong-scaling curves COINCIDE => the multi-GPU leg is
scripts/cluster/scaling_derecho/diagnosis.pbs-22-#     LATENCY-bound, not bandwidth-bound.  The lever is the per-step MESSAGE
scripts/cluster/scaling_derecho/diagnosis.pbs-23-#     COUNT, not the byte volume — so the collective census below is the
scripts/cluster/scaling_derecho/diagnosis.pbs-24-#     first-class signal, and it is STATIC (a CPU compile gives the same count,
scripts/cluster/scaling_derecho/diagnosis.pbs-25-#     but we capture it here on the real executable alongside the timings).
scripts/cluster/scaling_derecho/diagnosis.pbs:26:#   * Cube face-shard anti-scales at small tiles because the collective-permute
scripts/cluster/scaling_derecho/diagnosis.pbs-27-#     count grows with the shard count (~12 at 2 dev, ~41 at 6 dev on C24/L8)
scripts/cluster/scaling_derecho/diagnosis.pbs-28-#     while the per-message NCCL p2p latency (~30-80 us) does not amortise.
scripts/cluster/scaling_derecho/diagnosis.pbs-29-#
scripts/cluster/scaling_derecho/diagnosis.pbs-30-# Cube face-sharding requires the device count to DIVIDE 6 (1/2/3/6); a 4-GPU
scripts/cluster/scaling_derecho/diagnosis.pbs-31-# node therefore climbs the 1/2/3 ladder (4 is not face-divisible, and >6 uses
--
./packages/core/legoesm/parallel/cubesphere_exchange.py-1843-        raise RuntimeError(
./packages/core/legoesm/parallel/cubesphere_exchange.py-1844-            f"{context}: compiled HLO contains {len(bad)} all-gather "
./packages/core/legoesm/parallel/cubesphere_exchange.py-1845-            f"op(s) with full-cube face extent {bad[:8]} — the SPMD halo "
./packages/core/legoesm/parallel/cubesphere_exchange.py-1846-            f"is materializing all {n_faces} faces per device (compute "
./packages/core/legoesm/parallel/cubesphere_exchange.py-1847-            f"replication, HLO probe job 8456476).  Expected the "
./packages/core/legoesm/parallel/cubesphere_exchange.py:1848:            f"ppermute multiface exchange (collective-permute only).  "
./packages/core/legoesm/parallel/cubesphere_exchange.py-1849-            f"If the all_gather diagnostic backend was intended, set "
./packages/core/legoesm/parallel/cubesphere_exchange.py-1850-            f"LEGOESM_SPMD_FORCE_ALLGATHER=1 explicitly."
./packages/core/legoesm/parallel/cubesphere_exchange.py-1851-        )
./packages/core/legoesm/parallel/cubesphere_exchange.py-1852-
./packages/core/legoesm/parallel/cubesphere_exchange.py-1853-
--
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-36-#   Lane A: plain `legoesm-gpu` (README Step 1) is enough.
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-37-#   Lane B: `legoesm-gpu` WITH the route-A overlay (README Step 1b).
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-38-# The job runs lane A by default; enable lanes via qsub -v:
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-39-#   RUN_NCCL=1 (default 1)   RUN_ROUTEA=1 (default 0 — needs overlay)
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-40-#   RUN_TUNE=1 (default 0)   — lane T comm-tuning A/B ladder (fused-halo /
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:41:#     XLA collective-permute combining + pipelined p2p / PGLE) on the latlon
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-42-#     + cube lanes; outputs under _ab_tuning/ (excluded from the curves).
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-43-#     Run it AFTER the main lanes are green on this machine; see
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-44-#     docs/performance/scaling/spmd_message_census_2026-07-08.md.
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-45-#
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-46-# SUBMIT:
--
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-84-NLEV="${NLEV:-26}"
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-85-OC_NLEV="${OC_NLEV:-20}"
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-86-N_WARMUP="${N_WARMUP:-3}"
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-87-N_TIMING="${N_TIMING:-30}"
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-88-
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:89:# XLA collective-permute combining + pipelined p2p. #1113 found the route-B MPAS
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-90-# (lane E) throughput wall is the ppermute ROUND COUNT (9->24->30 as the SFC
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-91-# partition gains edge-coloring rounds) times a fixed ~0.11 ms/round launch
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-92-# floor — NOT transport bandwidth (GDR-off was immaterial at these message
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:93:# sizes). These flags fuse adjacent collective-permutes and pipeline the p2p,
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-94-# attacking the round count directly (the issue's top-priority remedy). Shared
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-95-# with the lane-T A/B `xla` arm. On by default for lane E; MPAS_CP_COMBINE=0
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-96-# for a baseline arm.
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-97-_XLA_COMM_FLAGS="--xla_gpu_collective_permute_combine_threshold_bytes=33554432 --xla_gpu_enable_pipelined_p2p=true"
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-98-MPAS_CP_COMBINE="${MPAS_CP_COMBINE:-1}"  # lane E CP-combining default (#1113)
--
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-284-#   base  : lane-C/D env exactly as above (the control arm — re-run so every
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-285-#           arm shares one allocation/fabric state; NEVER compare to an
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-286-#           earlier job's numbers)
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-287-#   fused : LEGOESM_LATLON_SPMD_FUSED_HALO=1 (bit-identical multi-pad packing;
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-288-#           trace-level receipt: atm 41->29 CPs/step, ocean 25% fewer)
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:289:#   xla   : collective-permute combining + pipelined p2p — the cube/latlon
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-290-#           rounds are data-independent, so the GPU combiner can merge what
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-291-#           the SPMD partitioner emits separately (code round floor = 4)
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-292-#   pgle  : profile-guided latency estimation (recompiles after profiling
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-293-#           runs — bench-jit-safe, NOT for AOT jobs; that is why it is not a
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-294-#           backend.py default)
--
./docs/dev-notes/les_plane_turbulence_notes.md-114-the BL laminarises. The dt=0.2 burst shows the dycore *can* momentarily hold
./docs/dev-notes/les_plane_turbulence_notes.md-115-strong turbulence (wvar~0.23), so the limiter is sustained production vs
./docs/dev-notes/les_plane_turbulence_notes.md-116-dissipation, which is **resolution-dependent**. The jax-alfa oracle runs
./docs/dev-notes/les_plane_turbulence_notes.md-117-128³–384³ at dx≈2–8 m with a (near-non-dissipative) pseudo-spectral solver.
./docs/dev-notes/les_plane_turbulence_notes.md-118-
./docs/dev-notes/les_plane_turbulence_notes.md:119:**128³ / dx=10 m + reduced hyperdiff (0.0003) CONFIRMS the limit:** the seed
./docs/dev-notes/les_plane_turbulence_notes.md-120-decays identically (wvar 0.036 → 0.0001 in ~80 s sim → 0), the same as 24³ and
./docs/dev-notes/les_plane_turbulence_notes.md-121-64³. So it is **not resolution** — the limiter is the compressible scheme's
./docs/dev-notes/les_plane_turbulence_notes.md-122-numerical dissipation (semi-implicit acoustic off-centring + biharmonic hyperdiff
./docs/dev-notes/les_plane_turbulence_notes.md-123-needed for stability), which caps the EFFECTIVE Reynolds number below the
./docs/dev-notes/les_plane_turbulence_notes.md-124-turbulence-sustaining threshold: the log-shear flow is linearly stable at this
--
scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-20-PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-21-echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-22-echo "=== worktree ==="; git -C $WT log --oneline -1
scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-23-
scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-24-run_acc () {  # $1 = bottom_drag_scheme
scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch:25:  JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 GM_CONFIRM=0 $PY - "$1" <<'PYEOF' 2>&1 | grep -vE "FutureWarning|warnings.warn|XLA|cuda|CUDA|Plugin"
scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-26-import sys, dataclasses, numpy as np, jax
scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-27-jax.config.update("jax_enable_x64", True)
scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-28-scheme = sys.argv[1]
scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-29-from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-30-from legoesm.ocean.experiments.dino import (DINOConfig, create_dino_z_star,
--
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-14-# Stage A (single process, CPU virtual devices): the tiled-vs-serial PARITY
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-15-#   gate at the adapter tolerances — correctness receipt BEFORE any timing.
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-16-#   Virtual-CPU rows are auto-flagged in metadata and are NOT speedup rows.
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-17-# Stage B (multicontroller, one process per GPU, route-B NCCL): the timed
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-18-#   run.  The bench REFUSES to record a row whose compiled HLO shows a
scripts/cluster/scaling_derecho/cube_tiled_step.pbs:19:#   full-cube all-gather (replicated execution) or zero collective-permutes.
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-20-#
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-21-# The tiled step is the DYNAMICS-ONLY base cut (the adapter refuses configs
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-22-# with hyperdiff/div-damp/sponge-implicit/duogrid — see
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-23-# tiled_step_adapter._refuse).  Compare against the face-only 6-GPU lane
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-24-# (blocker1_cube_shardmap.pbs / run_levante --cs-spmd) for the <=6 leg.
--
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-75-export NCCL_SOCKET_IFNAME=hsn
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-76-# #921: the closed-loop generation adds ONE extra collective the single-shot
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-77-# lane lacks — the dry-mass fixer's GLOBAL all-reduce
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-78-# (tiled_production_cdgrid._tile_fix_ps_mass_delta:
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-79-# psum over ("face","tile_i","tile_j")) layered on top of the halo
scripts/cluster/scaling_derecho/cube_tiled_step.pbs:80:# collective-permutes (cubesphere_exchange ppermute over the SAME axes). Under
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-81-# XLA GPU defaults (latency-hiding scheduler ON, async collectives ON,
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-82-# nccl_comm_splitting ON) the all-reduce clique and the permute cliques init
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-83-# CONCURRENTLY and the background NCCL init thread can establish them in a
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-84-# different relative order per rank -> cyclic wait / comm-init deadlock (ranks
scripts/cluster/scaling_derecho/cube_tiled_step.pbs-85-# reach channel/P2P setup but log ZERO "Init COMPLETE"). The code fix
--
./packages/core/legoesm/parallel/sharded_dynamics.py-722-        logger.info("make_sharded_step: single-device mode, using plain JIT")
./packages/core/legoesm/parallel/sharded_dynamics.py-723-        return _SingleDeviceStep(model)
./packages/core/legoesm/parallel/sharded_dynamics.py-724-
./packages/core/legoesm/parallel/sharded_dynamics.py-725-    # Activate explicit SPMD halo exchange for face-sharded cubed-sphere.
./packages/core/legoesm/parallel/sharded_dynamics.py-726-    # This replaces implicit cross-shard reads with explicit
./packages/core/legoesm/parallel/sharded_dynamics.py:727:    # collective-permute rounds, producing much better XLA communication
./packages/core/legoesm/parallel/sharded_dynamics.py-728-    # patterns.
./packages/core/legoesm/parallel/sharded_dynamics.py-729-    #
./packages/core/legoesm/parallel/sharded_dynamics.py-730-    # Iter-49 generalised activation from "exactly 6 devices" to "any
./packages/core/legoesm/parallel/sharded_dynamics.py-731-    # divisor of 6" (1, 2, 3, 6) on the allgather kernels; the
./packages/core/legoesm/parallel/sharded_dynamics.py-732-    # ppermute-multiface refit then made ppermute the DEFAULT exchange
--
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-86-``--sponge``/#836 was latlon-only and a silent no-op on MPAS).  Dry runs
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-87-survive because moist convection is the wave source.
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-88-
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-89-FIX ATTEMPT (sponge): ``_run_mpas`` now applies the #836 sin²-profile Rayleigh
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-90-decay to the MPAS edge winds as a config-gated post-step multiply
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md:91:(``--sponge``).  REFUTED as the cure: sponged runs die EARLIER (day 8-9 vs
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-92-9-14) — the sponge perturbs the trajectory but the instability is elsewhere.
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-93-(Kept as an option; a lid sponge is standard practice regardless.)
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-94-
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-95-## ROOT CAUSE LOCALIZED (2026-07-23 02:3x): moist-orographic 2Δσ checkerboard
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-96-## at ONE cell, amplified to global NaN by the GLOBAL conservation fixer
--
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-128-super-saturation pooling at the checkerboard hot levels deserves its own
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-129-bound/diagnosis, (c) revisit the vert4 default for moist MPAS configs.
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-130-
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-131-## Codex adversarial review verdict (2026-07-23 ~06:30, gpt-5.6-terra xhigh)
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-132-
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md:133:**F1 (CRITICAL, CONFIRMED): the MPAS ERA5 branch never attached ERA5 moisture
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-134-to the prognostic state** — it wrote `self.tracers["q_v"]` (a dead driver
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-135-dict) while the dycore advects `state.tracers`, still holding the moist-init
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-136-RH taper `rh_init·q_sat(seed_T, seed_p_s)·σ²`.  The spectral branch has the
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-137-re-attach; MPAS was missing it.  This RESOLVES the "seed p_s leaks into ERA5
./docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md-138-runs" mystery: seed q_v scales with q_sat(seed p_s), so the seed-p_s choice
--
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-904-
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-905-  **Disabling the cube sponge made the cross-grid
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-906-  disagreement WORSE**, not better.  This validates the
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-907-  iter-15 NOTE (the empirical 1-hour τ was tuned to
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-908-  partially compensate for an underlying imbalance) and
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md:909:  REFUTES the iter-64 hypothesis.
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-910-
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-911-  Combined finding from iter-63+64: at 3-day windows, the
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-912-  latlon-warm pattern is **robust to all tested cube
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-913-  damping reductions** — halving biharmonic+div_damp
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-914-  (iter-63) or disabling sponge (iter-64) does not pull
--
./tests/unit/test_tropopause_refined_sigma.py-292-
./tests/unit/test_tropopause_refined_sigma.py-293-    def test_hybrid_needs_the_true_dp_not_dA_plus_dB(self):
./tests/unit/test_tropopause_refined_sigma.py-294-        """The hybrid coordinate's own ``dsigma`` (= dA + dB) is the layer
./tests/unit/test_tropopause_refined_sigma.py-295-        thickness ONLY at p_s = p_ref, so using it as the mass weight leaves
./tests/unit/test_tropopause_refined_sigma.py-296-        the filter conservative only to the p_s/p_ref departure.  Pins that
./tests/unit/test_tropopause_refined_sigma.py:297:        the call site passes the real ``dp`` (codex round 2 CONFIRMED-BUG)."""
./tests/unit/test_tropopause_refined_sigma.py-298-        hy = make_hybrid_levels(40, 200.0, stretching=2.0)
./tests/unit/test_tropopause_refined_sigma.py-299-        dA = np.asarray(hy.dA, np.float64)
./tests/unit/test_tropopause_refined_sigma.py-300-        dB = np.asarray(hy.dB, np.float64)
./tests/unit/test_tropopause_refined_sigma.py-301-        p_s = 62000.0                                   # plateau column
./tests/unit/test_tropopause_refined_sigma.py-302-        dp_true = dA * float(constants.p_ref) + dB * p_s
--
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-192-miniature. Checked on the source geometry alone, so `polar_fill=True` can never
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-193-pass where `polar_fill=False` fails. Real sources are far inside it (CORE-II 0.51°,
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-194-JRA55-do 0.15°); a ±10° regional band, or a latitude axis misread as radians
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-195-(±1.57°), is refused.
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-196-
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md:197:## CONFIRMED downstream: this was the ½° latlon polar blow-up
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-198-
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-199-The zero-forcing regime was not hypothetical — it was crashing a real configuration,
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-200-and the crash had been worked around rather than diagnosed.
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-201-
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-202-`scripts/cluster/omip_nemo/_diag_llh_polarcap.sbatch` records a half-degree
--
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-229-Arm B is 70× past the step-3 failure point with no polar excursion, so
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-230-**`--mask-polar-cap-lat` is no longer required for this configuration.** Eight
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-231-launchers use `--latlon-res 360x720`; all were exposed.
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-232-
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-233-Scope of the claim, deliberately narrow:
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md:234:- **CONFIRMED** — the ½° polar NaN/blow-up at lat 89.75 was caused by the
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-235-  zero-coverage polar row, and is fixed.
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-236-- **NOT claimed** — the separate first-step HANG investigated in
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-237-  `_diag_llh_hang.sbatch` (`TotalCPU=0`, a GPU/sync deadlock) is a different
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-238-  symptom and was not tested here.
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md:239:- **REFUTED** — an earlier guess of mine that this bore on the Arctic halocline /
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-240-  excess-ice-growth work. It cannot: `_conservative_regrid_to_latlon` is reached
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-241-  only for `latlon`/`latlon_regional`, while the tripole and MPAS runs use
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-242-  `_nn_interp_to_points`, untouched by any of this. At 1° the affected row is also
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-243-  only 0.26% of Arctic ice area.
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-244-
--
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-262-**The HANG is a separate matter and remains unexplained.** `_diag_llh_hang.sbatch`
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-263-described a first-step *hang* (`TotalCPU=0`, a GPU/sync deadlock or stalled CUDA
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-264-alloc), and it did **not reproduce in any arm** — `min`, `pf` and `host` all
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-265-completed with `rc=0`, and even the pre-fix arm NaN'd rather than hung. So:
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-266-
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md:267:- **CONFIRMED** — the ½° NaN/blow-up at step 3, in both the reduced and the full
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-268-  `host` configuration, was the zero-coverage polar row; it is fixed.
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-269-- **NOT reproduced, NOT attributed** — the `TotalCPU=0` hang. Nothing here explains
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-270-  it; it may have been transient (node/driver) or fixed by something unrelated.
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-271-  Do not credit this change with it.
./docs/dev-notes/regrid_polar_coverage_2026-07-24.md-272-
--
./docs/dev-notes/parameterization_checks.md-44-- **33. YSU nonlocal countergradient (i44, fresh audit):** YSU was MISSING its defining nonlocal term — only the local K-profile + entrainment-K were present, so the convective BL was not transported counter-gradient (the whole point of YSU vs a local scheme). Added `γ_c = b·max((w'θ')₀,0)/(w_s·h)` [K/m] (Troen-Mahrt 1986 / Hong et al. 2006), config `countergrad_coeff=6.5`, unstable-gated (zero for neutral/stable), `w_s=w*` convective scale (⇒ the usual `(w'θ')^{2/3}` scaling), applied as a heat-surface-flux enhancement (same convention as HB). Verified: active for unstable, gated off for stable (ON==OFF), AD-safe; 105 turbulence tests pass. *(Uses the codebase's HB-style surface-flux-enhancement; a per-level γ_c flux would be a further refinement.)* **Also added the unstable gate `max(w'θ',0)` to the HB countergradient (#32) for consistency — the countergradient is a convective-only feature.**
./docs/dev-notes/parameterization_checks.md-45-**34. REGRESSION RESCUE (i46) — three faithful upgrades were SILENTLY REVERTED in the working tree (ledger ✅ was STALE):** `turbulence/config.py` had lost the Smagorinsky (#26) + YSU (#33) config fields and BOTH the LouisConfig field AND `louis.py` source for #27. Symptoms: `smagorinsky` & `ysu` **crashed at runtime** (`AttributeError: 'SmagorinskyConfig' object has no attribute 'l_mix_max'`; `'YSUConfig'…'countergrad_coeff'`); `louis` had reverted to the simplified `f_h=f_m`. Restored all three: `SmagorinskyConfig` `Km`→`C_s=0.2`+`l_mix_max=100`; `YSUConfig.countergrad_coeff=6.5`; `LouisConfig.b_heat_ratio=1.5` + re-implemented louis.py LTG82 separate heat function (momentum f_m **bit-identical** via shared denominators, `ratio=1`→old). Verified: full turbulence 56✓, diff-atmosphere+diff-params 60✓ (orig 5 failures cleared). **Also fixed 2 stale combined-physics tests** (`test_turbulence_only`, `test_scheme_selection_turbulence`): they asserted nonzero dT_dt / scheme-divergence on the quiescent `held_suarez_init` column (near-isothermal ⇒ stable), but the faithful Smagorinsky–Lilly *correctly* gives ~zero interior K there (hard Lilly cutoff) — the assertions assumed the OLD unconditional-mixing constant-Km. Added DRY `_make_turbulent_setup()` (sheared jet + super-adiabatic surface) reused by both ⇒ 23✓. **Codex adversarial review (gpt-5.5, available again this session):** flagged positional-NamedTuple-shift hazards (#1-3) — verified NOT real (repo grep: ZERO positional construction of these 3 configs, all keyword/`._replace`); and `b_heat_ratio<0`→singular K_h (#4) — equivalent to pre-existing unconstrained `b_louis`/`d_louis`, default 1.5 safe, hardened via docstring positivity note. **LESSON: re-verify ledger ✅ claims against the actual tree (`pytest` per family) — do not trust the ledger alone; ≥4 items had drifted.**
./docs/dev-notes/parameterization_checks.md-46-**35. WHOLE-SUITE DRIFT SWEEP (i47) — applied the i46 "re-verify against tree" lesson family-wide.** Ran per-family pytest (atmosphere / ice+land+coupler / ocean-params) to catch crashes & stale tests like i46's. **Results:** ice+land+coupler **410✓** (clean); atmosphere **541✓** after fixing 2 stale tests; ocean-params **281✓** (clean) — ~1232 param tests green total. **(a) DCA `test_moist_static_energy_conservation`:** stale test — `_adjust_one_iteration` gained a required `instability_blend_sharpness` arg (config-driven) but the direct call didn't pass it (`TypeError`); fixed the call to pass `DCAConfig().instability_blend_sharpness`. Scheme correct. **(b) GWD `test_gwd_grad_through_T[hines]`:** grad-through-T was exactly 0.0 — traced to the per-level drag clip `clip((σ_grown²−σ_sat²)·ρ, 0, Fmax)` hitting `Fmax=0.1` at *every active level* of the coarse strong-shear test column, pinning drag at the constant cap ⇒ T-path (via `σ_sat=N/m_*`) severed. Fixed the test helper to raise `Fmax` (so the Doppler-spread term, which carries the T-dependence, is the binding one). 73 GWD tests pass. **This surfaced a real production limitation → logged as F-GWD-2 (codex-confirmed DOCUMENT-AND-DEFER).** **Hand-audits this sweep (all clean — no fix):** emanuel `_mixture_buoyancy` (Emanuel-91 buoyancy sorting: linear mix + one-step sat-adjust clip[−q_c,q_v] AD-safe, condensate loading `−T·q_c`, sign-crossing B(χ), denom `1+L/c_p·dq_s/dT>1`); surface_layer (drag/SH/LH signs, wind floor, validate guard); soil_hydraulics (VG-Mualem where-before-pow K(ψ=0)=K_sat, VG_C floored, all dispatchers raise on unknown); rheology (Hibler P, VP constitutive signs, EVP/mEVP convex relaxation, Δ sqrt-arg floor, mEVP α≥1 guard); bulk_flux MOST (reciprocal inv_L neutral-AD, LY09 C_DN+Stanton/Dalton); tiedtke (CAPE closure `ρ_BL·CAPE⁺/(g·τ)` units, M_b≥0 bounded, convex M_u relax, downdraft mass-conserving rain re-evap); KPP/LMD94 (B_f>0=unstable convention consistent, L_MO/ζ signs, w_m¼/w_s½⇒Pr_t<1, d_eff surface-layer cap, G=σ(1−σ)², interior K_0(1−(Ri/Ri_0)²)³, **nonlocal γ in conserving flux-form**, dz_safe AD, separate K_bg/A_bg — codex KPP review TIMED OUT 480s xhigh+websearch, hand-audit substituted); zhang_mcfarlane (CAPE closure `ρ_BL·CAPE⁺/(g·τ)` #15-units, convex M_b relax, bounded — shares F-CONV-MSE kernel); mynn25/NN09-2.5 (L_S/L_T/L_B harmonic master length, level-2 derived consts App-A, flux-Ri disc-clip AD, alpha_c eq42, SM25/SH25 eqs27-28 all degeneracies floored, w_thv 1/ε−1 #16, Obukhov masked-divisor #7, G_M>0/G_H signs, K=L·q·S>0, K_q=3K_m eq67); ocean backscatter (energy-injection sign correctly ADDED vs dissipative-subtract; reservoir `dE=η·ε_diss−ε_bs−E/τ` with `η=0.9<1` ⇒ sources only from resolved dissipation, hard-clamped `E∈[0,0.1]` ⇒ cannot blow up; √(max(E,0)) AD-safe; default-OFF); coupler tile_fractions (f_ocean+f_ice+f_land+f_lake≡1 algebraically, all ≥0, pure-ocean reciprocal floor #9 AD-safe, conservative area-weighted blend); ice dynamics (momentum `m du/dt=τ_a+τ_o−mfk×u+∇·σ`, `+div(σ)` sign correct, drag signs, Coriolis implicit, A-factor cancels). **`scripts/validate_convection_physics.py` run (i47): VALIDATION PASSED — all 4 enforced tests clean across all 9 schemes** (sign H≥0/Q_v≤0/Q_c≥0; stable-sounding gating; magnitudes within 100× of SBM). Test-1 (diagnostic, unenforced) quantifies F-CONV-MSE as **column-dependent**: mass_flux closed-column MSE residual = **8.8 %** on the tropical sounding (under the 10 % bar) vs ~64 % on `_make_unstable_columns` (the spec column); emanuel 53 %/kain_fritsch 41 % (large condensate detrainment). Confirms the convection family is sign/gating/magnitude-sound; the residual is the known F-CONV-MSE kernel issue, worst on specific entrainment regimes. **`.physics-validator/ocean/diff_probes.py` run (i47): 23/23 pass** (EOS dρ/dT,dS,dp + α>0 sign; hydrostatic-p; sponge; freshwater virtual-salt sign; PE-iterate grad; implicit-vert-diffusion no-flux conservation 2e-16; bottom-drag positivity; NPZD N conservation 3e-16; KPP finite+nonlocal+grad). **Found+fixed a WRONG probe expectation** (#36): `plume_convection` surface-tendency-==0 check assumed a penetrative-plume-detrains-below model, but the scheme is **heat-conserving convective adjustment** — on the probe's statically-unstable column (cold/dense T=2°C surface over warm T=14°C deep) it correctly **warms the surface, cools the deep, conserves column heat to machine precision (resid/gross=0)**; rewrote the probe to assert that (a zero surface tendency would itself be the bug). Ocean plume convection scheme verified CORRECT. **F-CONV-MSE confirmed correctly deferred:** default convection=`sbm` (conserves) repo-wide; the flux-form fix `(1/ρ)∂_z[M(X_u−X)]` (replacing the gradient form's fixed-δ₀ detrainment that breaks telescoping) is well-specified but CHANGES the heating/drying *profile* ⇒ needs RCE/SCM climate validation before shipping to the 4 opt-in mass-flux schemes (per "be precise, operational"). **Verified #28-32 source signatures all present** (not reverted like the i46 turbulence configs). **`validate_strict` membership lists** for convection/turbulence/gwd **match the integration.py factories exactly** + each factory raises on unknown (CLAUDE.md "GAP" note is stale — already resolved). **AD-trap re-sweep (i47, repo-wide `jnp.sqrt` grep over atmosphere/ocean/ice/land/coupler physics): CLEAN** — every bare sqrt verified safe (tke/clubb floored `max(·,tke_min)` upstream; smag/cloud double-`where`; mynn25/ice_shelf/stomata `+eps²`; holtslag/ysu/bulk_flux arg≥1; wind sqrts `+U_min²` with `U_min=1.0`; backscatter/dynamics positive grid metrics) — no drift reintroduced the iter-2/3 sqrt'(0) traps. **ice/shortwave delta_eddington penetration (minor, opt-in; default ice-SW=`constant`):** `i0_vis=0.70` applied as fraction of *incident* SW (no in-ice Beer-Lambert decay) vs CICE's `I0·(1−α)·exp(−κ_ice·h)` ⇒ surface/ocean partition biased toward ocean for thick bare ice; **column energy still conserved** (`absorbed=(1−α)F−penetrated`). Disclosed in `shortwave.py` docstring; faithful upgrade needs CICE `I0` convention + `κ_ice` reference + validation (not guessed).
./docs/dev-notes/parameterization_checks.md-47-**Verified-clean:** microphysics(safe_pow; all schemes i35); ocean lateral/surface/BGC/ice-shelf/mixing/mpas; turbulence tke/ysu/edmf/clubb/HB; GWD lindzen/mcfarlane/rayleigh; radiation; land carbon/stomata/snow; lake. Constants clean. Coeff spot-checks COARE3(i9)/Kessler(i13)/Louis-f_m(i15). **Self-review i38-40:** F-OCEAN-1/emanuel/smagorinsky/Louis/bulk_flux-MOST re-reviewed — all sound (1 gap found+fixed = i38 surface-layer cap). **Deep ref-audits i42 (no fix — all faithful):** ocean SW penetration (Jerlov R/ζ1/ζ2 match Paulson-Simpson 1977 Table 1; bottom-leak conserved; units K/s) · gray radiation (Frierson τ-form; Schwarzschild two-stream; heating-rate sign verified: emit-to-space ⇒ cooling) · solar insolation (declination via DEG_TO_RAD; exact daily-mean Q formula; AD-safe arccos). Sea-ice albedo = honestly-disclosed smooth CCSM3 approximation (0.7 cold/0.5 melt bare, sqrt thin-ice ramp) — not bit-exact but sound, not a defect. **Deep ref-audit i45 (no fix — faithful):** ocean GM/Redi lateral mixing — slope `S=−∇_hρ/∂_zρ` (∂_zρ guarded <0 stable); Griffies-1998 small-slope tensor with correct combined coeffs `(κ_R−κ_GM)` horizontal / `(κ_R+κ_GM)` vertical (symmetric Redi + antisymmetric GM skew); DM95 tanh slope taper; vertical & horizontal flux-divergence signs consistent ⇒ positive-definite (diffusion, not anti-diffusion); Visbeck-1997 `κ=α·l²·σ` with `σ=N|S|` (= Eady growth rate via thermal wind) + Rossby-radius `l=N̄H/f`.
./docs/dev-notes/parameterization_checks.md-48-
./docs/dev-notes/parameterization_checks.md:49:**37. Ocean energy-backscatter OPERATOR (oracle bug 3) — the i47 sign-only audit above ("energy-injection sign correctly ADDED") MISSED the operator ORDER.** Jansen & Held (2014) / Bachman (2019) deterministic backscatter is a NEGATIVE HARMONIC (Laplacian) viscosity `-ν_bs∇²u` (growth ∝ +k²), stabilised by a paired scale-selective biharmonic/Leith dissipation + the `E_max` cap. The code instead returned an ANTI-BIHARMONIC `+∇²(ν_bs∇²u)` (growth ∝ +k⁴) — amplifying the GRID scale MOST (docstring defended it with backwards physics). BOTH forms inject energy (same sign) so the sign-only audit passed; only the spectral character was wrong. **Fixed** (both grids): single (negated) harmonic stress-divergence, coeff `ν_bs=c_bs·Δ·√E` [m²/s] (MPAS Δ³→Δ ⇒ `c_bs` now grid-consistent); injection sign + AD-safety preserved; honest docstring (unstable alone → needs paired biharmonic dissipation; opt-in, NOT yet wired to production). Exact negative-Laplacian regression locks (cgrid+MPAS) falsify the biharmonic two-pass. Codex adversarial review: physics/sign/units/AD/tests all CONFIRMED-CORRECT (1 doc defect found+fixed). **LESSON: a sign-only energy audit cannot certify a backscatter — check the SPECTRAL order (k² vs k⁴).** Screenshot oracle bugs 1,2,4,5,6 (kuo icond2-conservation, edmf flux-form, kpp V_t √βT, harmonic A_h/K_h AD-guard, mle float(ce)) were VERIFIED already-fixed in HEAD (2026-06-29 oracle commits) — that review had run on a pre-fix tree.
./docs/dev-notes/parameterization_checks.md-50-
./docs/dev-notes/parameterization_checks.md-51----
./docs/dev-notes/parameterization_checks.md-52-
./docs/dev-notes/parameterization_checks.md-53-## Open findings (deferred — all need user-side prerequisites)
./docs/dev-notes/parameterization_checks.md-54-- **F-GWD-1 ✅ RESOLVED (2026-07, fix/atm-param-oracle-review).** Deposition now carries the smooth `tanh((c−U_proj)/w)` directional factor (`w = PrognosticSpectralConfig.direction_sign_width`, excluded numerics width), the carried flux is clamped non-growing (`F_new = min(blend, F_carry)`, kills the sigmoid-tail overshoot that made `drag<0`), and the heating/`eps_gwd` use the **intrinsic form** `g·drag·(c−U_proj)·tanh((c−U_proj)/w) ≥ 0` (E3SM `dttke` analogue) — so `eps_gwd ≥ 0` **by construction** even when fast waves accelerate the flow toward `c` (the i6 objection: with the mean-flow-KE form the sign factor alone was insufficient; the intrinsic form resolves it). Tests: `test_gwd_physical_realism.py::test_prognostic_spectral_{slow_spectrum_drag_opposes_flow,heating_nonneg,symmetric_two_wave}` (former strict-xfails replaced). *Remaining user follow-up: QBO/momentum-deposition benchmark before default-on; scheme stays opt-in (default GWD=`none`) and in the RCE-realism TODO.*
--
./docs/dev-notes/issues/clubb_parcel_lscale_vs_shared_mixing_length.md-8-vs CLUBB-JAX): an entraining parcel ascends/descends until its buoyancy is
./docs/dev-notes/issues/clubb_parcel_lscale_vs_shared_mixing_length.md-9-exhausted; `Lscale_up`/`Lscale_down` are combined geometrically.
./docs/dev-notes/issues/clubb_parcel_lscale_vs_shared_mixing_length.md-10-
./docs/dev-notes/issues/clubb_parcel_lscale_vs_shared_mixing_length.md-11-`atmosphere/physics/_shared.py:mixing_length` (used by `clubb_lite` and
./docs/dev-notes/issues/clubb_parcel_lscale_vs_shared_mixing_length.md-12-others) is a simple Blackadar-style local length. The two were compared
./docs/dev-notes/issues/clubb_parcel_lscale_vs_shared_mixing_length.md:13:during condensation and CONFIRMED different numerics, so the parcel version
./docs/dev-notes/issues/clubb_parcel_lscale_vs_shared_mixing_length.md-14-stays inside `clubb.py` per the one-file-per-scheme convention.
./docs/dev-notes/issues/clubb_parcel_lscale_vs_shared_mixing_length.md-15-
./docs/dev-notes/issues/clubb_parcel_lscale_vs_shared_mixing_length.md-16-**Idea:** the parcel buoyant-sorting length is physically richer (nonlocal,
./docs/dev-notes/issues/clubb_parcel_lscale_vs_shared_mixing_length.md-17-stability-aware) and could eventually supplant the `_shared` Blackadar length
./docs/dev-notes/issues/clubb_parcel_lscale_vs_shared_mixing_length.md-18-for other schemes. That would mean promoting it OUT of `clubb.py` into
--
./packages/core/legoesm/parallel/tiled_production_cdgrid.py-2447-    init cannot deadlock (issue #921).
./packages/core/legoesm/parallel/tiled_production_cdgrid.py-2448-
./packages/core/legoesm/parallel/tiled_production_cdgrid.py-2449-    The closed-loop blocked step
./packages/core/legoesm/parallel/tiled_production_cdgrid.py-2450-    (:func:`make_tiled_fv3_hydrostatic_step_blocked_2d` with ``fix_mass=True``,
./packages/core/legoesm/parallel/tiled_production_cdgrid.py-2451-    and the operator-split twin) issues, inside ONE compiled executable, BOTH
./packages/core/legoesm/parallel/tiled_production_cdgrid.py:2452:    the halo collective-permutes (``jax.lax.ppermute`` over the
./packages/core/legoesm/parallel/tiled_production_cdgrid.py-2453-    ``(face, tile_i, tile_j)`` mesh axes — the tiled halo cliques) AND the
./packages/core/legoesm/parallel/tiled_production_cdgrid.py-2454-    mass-fixer GLOBAL reduction (``jax.lax.psum`` over the SAME axes —
./packages/core/legoesm/parallel/tiled_production_cdgrid.py-2455-    :func:`_tile_fix_ps_mass_delta` / :func:`_tile_fix_ps_mass_target`).  NCCL
./packages/core/legoesm/parallel/tiled_production_cdgrid.py-2456-    communicator init is itself a collective over the clique; under the XLA/GPU
./packages/core/legoesm/parallel/tiled_production_cdgrid.py-2457-    defaults (latency-hiding scheduler + async collectives +
--
./docs/COMPARE_REANALYSIS.md-1-# Compare-to-Reanalysis + LES-Informed Column Correction
./docs/COMPARE_REANALYSIS.md-2-
./docs/COMPARE_REANALYSIS.md-3-**Branch:** `feat/compare-reanalysis`
./docs/COMPARE_REANALYSIS.md:4:**Status (compressed at iters 50 + 60 + 70 + 80 + 90 + 100 + 110 + 120 + 130 + 140 + 150 + 160 + 170 + 180 + 190 + 200 + 210 + 220 + 230 + 240 + 250 + 260 + 270 + 280 + 290 + 300 + 310 + 320 + 330 + 340 + 350 + 360 + 370 + 380 + 390 + 400 + 410 + 440 + 450 + 460 + 470 + 480 + 490 + 500 + 510 + 520):** The whole offline loop is built, tested + Codex-reviewed, and **structurally complete** end-to-end — every stage `run (AMIP/CMIP) → time-mean → compare to ERA5 → rank → cluster → LES-diagnose → feedback → inject (turbulence_override) → RE-RUN` is real and tested (capstone iter 37; the injection provably changes the simulation, iter 35; feedback↔physics column ordering locked, iter 36). A **runnable HPC entry point** exists — `scripts/run/run_correction_campaign.py` (`make_base_driver_builder` for AMIP **and** CMIP, real-ERA5 reference loading, per-column `clubb_lite` C_K/Pr_t/C_eps correction (single or SIMULTANEOUS multi-coefficient), per-round bias report, self-describing summary+health verdict). The campaign's saved JSON now **deploys back into a fresh production run** via `legoesm.training.deploy_correction.corrected_turbulence_override` (iter 57, grid-identity-guarded iter 58) — the literal "update the parameters in the AMIP/CMIP simulation" plumbing is closed and `validate_strict`-checked. A **cross-resolution (grid-AGNOSTIC) deploy** path also exists end-to-end: a campaign run with `--feedback-strategy environment` now EXPORTS a `<out>.env_kernel.json` (the RAW env→coefficient regression, iters 69–70 producer wiring) that `apply_env_kernel_override` re-evaluates on ANY target grid by environmental similarity (so a cheap low-res campaign deploys on an expensive high-res run), with a coverage/in-hull domain-shift diagnostic. A **perfect-model (identical-twin) OSSE harness** (`legoesm.training.perfect_model_osse`, iters 59–60) demonstrates clause 5's LOGIC in a controlled twin — the model's own run with a KNOWN coefficient is the pseudo-truth, and the loop is checked to LOWER the bias AND RECOVER the known parameter (single AND simultaneous multi-coefficient C_K+Pr_t+C_eps) — the cheap go/no-go an HPC user runs before the real-ERA5 campaign. A **cross-resolution OSSE** (iter 72, `run_cross_resolution_osse`) extends this to the grid-AGNOSTIC env-kernel deploy: it LEARNS the env→coefficient correction on a coarse grid and DEPLOYS it on a finer grid by environmental similarity, demonstrating (in the twin) that the kernel TRANSFERS and lowers the finer grid's bias — the first measured bias-reduction from the iter-69/70 deploy, gated by an out-of-hull (extrapolation) trust check. The per-iteration table below records each stage's module + tests (compressed at iters 10/20/30/40/50/60/70/80, folding iters 1–50 into summary rows; full detail in git log). **Remaining (the done-criterion):** ONLY the EMPIRICAL demonstration — execute the multi-day campaign on real ERA5 at HPC scale and observe the LES-informed coefficients *lower* the bias (not unit-testable here, not honestly fakeable; the result is genuinely unknown until run). **Breadth — ALL DONE through iter 90:** the LES spin-off works on ALL 4 grid families (lat-lon, cubed-sphere, MPAS/Voronoi, Gaussian/spectral — extractors + compare-side handoffs + capstones, iters 73–85); distributed/MPI FEEDBACK sharding is real-mpirun-validated on all 3 structured families (iters 61–64); and the **distributed-MPAS LES-correction campaign now runs END-TO-END under real MPI** (owned-cell mask 86 + cross-rank global top-k 87 + collective correction loop 88 + `build_distributed_correction_campaign` 89, all `mpirun -np 2`-validated). Geostrophic LES-forcing wind is now supplied on ALL 4 grid families — lat-lon + Voronoi (iter 82) + Gaussian (iter 90, via the shared `spectral_gradient_3d`) + cubed-sphere (iter 93, the metric-correct 2×2 basis solve `_gradient_cubed_geographic_3d`, CONVERGING incl. panel edges after iters 91/92's naive attempts were Codex-rejected). The one-shot CLI spectral-CHECKPOINT restart load is DONE (iter 92, `load_restart` reconstructs the spectral keys via `reconstruct_spectral_state_from_npz`). The distributed-MPAS campaign is runnable for SINGLE + simultaneous MULTI-coefficient correction (iters 89/95) AND now has a ONE-CALL runnable entry point `build_distributed_mpas_campaign` (iter 96) that does the MPI-aware setup (partition the global mesh → wire the rank-local mesh as grid + driver) so an HPC user supplies only the global mesh + a `build_local_driver`. The distributed slice now FAILS LOUDLY on a global/reference/area mesh-count mismatch (iter 97 — exact `nCells_global` guards replacing a JAX gather's silent OOB-clamp), and a DEFAULT-ON collective pre-flight (`assert_partition_covers_global`, iter 98) verifies the rank partition covers the global mesh EXACTLY once (no silently-uncorrected or double-counted cell) before a multi-day run. A DEFAULT-ON physical-plausibility pre-flight (`validate_reference_physical`, iter 99) catches a units/sign error in the loaded ERA5 reference (T/p_s/q_v wrong units) before it drives the loop to 'correct' a fake bias. The campaign now reports the LES-diagnosis VALIDITY rate (`n_diagnoses_valid` per round, `n_diagnoses_valid_total` in the summary) and a `no_valid_diagnoses` health verdict (iter 100), so a run that makes no correction because the spin-off LES never developed turbulence says so instead of looking like a broken correction; it also EARLY-ABORTS (collective-safe, `stop_reason="no_valid_diagnoses"`, default-on, CLI `--keep-dry-rounds` to disable) after `patience` consecutive dry rounds rather than burning multi-day HPC time (iter 101). The CLI launch chain (flag → parser → shared `_campaign_knobs_from_args` → builder → loop) is now fully unit-tested (iter 102), so a wiring regression fails in CI rather than on a multi-day HPC launch. The spin-off LES now FAILS FAST on an acoustically-unstable timestep (iter 103, `run_forced_les` horizontal acoustic-CFL pre-flight reusing `cfl_diagnostic`; all 3 LES CLIs default to the CFL-safe `--les-dt 0.5`) instead of blowing up partway through a multi-day run. A CI import-resolution guard (iter 104, `test_cli_imports_resolve.py`) now catches a stale function-scope import in any of the 4 compare-reanalysis HPC CLIs (the OSSE-class dead-entry-point bug) before a launch, and the campaign→deploy JSON contract is round-trip-tested (iter 105, `build_campaign_output_dict` → `corrected_clubb_config`) so the on-disk format that updates a production run cannot silently drift. The real-ERA5 loader now FAILS LOUDLY on a missing/misnamed required variable (iter 106, `load_era5_slice` required-raise + bidirectional `resolve_var`) instead of silently feeding zeros into the bias, and the full real ERA5-input→reference chain (load→regrid→interp→carry→ColumnState) is now exercised end-to-end by an integration test (iter 107) instead of being bypassed by the compare test's monkeypatch. A SEVERE ERA5→cubed-sphere regrid bug (the input weights were built from a Gaussian PROXY of the uniform lat-lon ERA5 grid, mis-indexing source cells by up to ~154° of latitude) was found + fixed (iter 109, `compute_latlon_to_cs_weights` from the ACTUAL source nodes; content-fingerprinted weight cache; T=lat regression test) — the cubed-sphere real-ERA5 reference is now spatially correct. A follow-up audit (iter 110) verified the OTHER three ERA5→grid input paths: spectral (`regrid_latlon_to_gaussian`, scipy from actual coords, descending-lat OK) and MPAS (`era5_to_mpas_carry`, never a proxy, spatial-fidelity tested) are CORRECT; the MPAS weight CACHE had the same iter-109 key weakness and was re-keyed onto the shared content-fingerprint (collision/GC-safe). All four ERA5→grid input paths are verified spatially correct with collision/GC-safe weight caches, and their specific-humidity→mixing-ratio conversion is now factored onto the single canonical `thermo` helper (iter 111, which also fixed a latent float32 divide-by-zero in that helper), and the most-shared input numeric `interp_pressure_to_sigma` is locked by a real numerical regression suite (iter 112, replacing a vacuous constant-field test; verify-first confirmed the routine is numerically correct). The clause-3 worst-column ranking's cross-variable commensurability is locked too (iter 113 — a mis-set per-variable normalization that hid a variable from selection now fails in CI; verify-first confirmed the comparison/metric/LES-inverse/deploy code is correct and well-tested), and the compare-reanalysis unit suite (clause-3 forcing, clause-4 closure, clause-5 OSSE, ranking metrics, promotable bounds) is now deterministic at fp32 (iters 114–115 — no longer silently dependent on a cross-file x64 leak; xdist/isolation-safe; iter 115 also verify-first-audited the full subsidence→warming forcing-chain sign as correct + tested at every link). The clause-5 worst-column bias diagnostic is hardened against a padded-index dilution foot-gun (iter 116, optional `valid` mask + documented precondition; verify-first confirmed `bias_metrics` correct and the campaign callers already safe via the manifest's invalid-slot drop). The geostrophic LES-forcing wind now supports the orographic surface-geopotential term over TERRAIN on all 4 grids (iter 117, optional `phis` through the helper + 4 extractors + dispatcher, Codex-approved physics; the deep driver→extractor `phis` wiring is the documented next step, production currently runs flat/unchanged). The iter-117 orographic term is now ACTIVATED end-to-end through the campaign library API (iter 118) AND symmetrically through the perfect-model OSSE go/no-go (iter 120), with a fail-loud `phis.shape==p_s.shape` guard (iter 119) — `phis` threaded through the builders → `make_les_diagnose_fn` → `process_column` → `extract_gcm_column` → extractor, Codex-verified at each step. The orographic feature's consistent-topography source is now a one-call helper `model_phis_from_driver(driver)` (iter 122 — returns `None` for a flat model, so passing it is always safe). Its flat-detection was then made honest (iter 125 — a real flat `ModelDriver` initialises `phis = zeros(...)`, not absent, so an all-zero field now maps to `None` via the shared `phis_or_none_if_flat`), and the orographic term is now **reachable from the campaign command line** (iter 126 — `--orographic-forcing {auto,on,off}`, default `auto`; `main()` resolves the model's OWN topography via the SIDE-EFFECT-FREE `ModelDriver.static_topography_phis()` probe — no phantom run dir — and threads `phis` into both the single- and multi-coefficient campaigns; `on` fails loud on a flat model). **Iters 127–149 (compressed rows below)** added PER-VARIABLE bias reporting + cumulative resume-consistent trajectories (130–139), the TIME-MEAN ERA5 climatology reference `--era5-n-times` + ERA5/CLI input fail-loud hardening (140–143), real-coupled CMIP entry-point coverage (144), the closure-diagnosis STRETCHED-grid + AD-safety locks for the diagnosed K/`K_m`→`C_K` (145–146), an opt-in fail-loud on a NO-OP cross-grid deploy (147), and a fail-loud + cross-dispatch lock around the LES surface-BC contract (148–150). The in-repo clause-6 ("updating params IMPROVES the bias") LOGIC is proven in controlled twins (single + simultaneous multi-coefficient + cross-resolution OSSE, all asserting genuine bias reduction + parameter recovery). **Iters 150–160 (compressed row below)** BUILT the surface-flux LES `prescribe="fluxes"` path (iter 151) — a prescribed kinematic θ/q_v flux injected on the LES surface cell, real-compressible-Euler-dycore-validated (warms+moistens the surface, iters 156–157) — plus a codex-blocked-period (Jun 19→24) campaign of verify-first audits + untested-helper/invariant locks that found NO correctness gaps in the core pipeline (gate, line-search, deploy, compose, manifest, clustering, bias_metrics all confirmed correct + tested). **The ONLY remaining done-criterion item is the empirical real-ERA5 HPC demonstration (a multi-day run on real ERA5 at HPC scale, observing the LES-informed coefficients lower the bias — not unit-testable here, not honestly fakeable). TRACKED follow-ups (not blocking the demo): the two surface-flux production changes (151/154) await the MANDATORY codex adversarial review when the usage cap resets (Jun 24); the surface-TEMPERATURE (`prescribe="T_s"`) bulk-flux path + wiring an extractor to SET `prescribe="fluxes"` from the GCM-column SST (so the CAMPAIGN uses surface fluxes); and the cheap `--dry-run` pre-flight.** **Iters 290–299:** the cross-resolution OSSE go/no-go now has an OPERATOR CLI — `run_perfect_model_osse.py --fine-resolution N` learns the env-kernel on a coarse grid and validates its TRANSFER to a fine grid in a twin (built 296, input-hardened 297, exit-code-gated 298); a no-duplication sweep (291–295) factored every single/multi builder family into one place and en route FOUND + FIXED a real OSSE-CMIP crash (raw `--coupled-preset` name vs resolved preset, 295); plus a runbook automated-gating section (290) and an adversarial self-review confirming `bias_metrics` + `column_clustering` correct + fully tested (299). **Iters 300–309 (codex-capped Jun 19→24):** a robustness + operator-runnability campaign — model-divergence guards on the baseline + pseudo-truth (301/302), fail-loud/clean deploy hardening across every malformed-artifact class (303/304/309), the operator path to the empirical run smoothed (ERA5 loader all-missing-vars 306, config-gen toy-resolution warning 307, sbatch↔CLI flag lock 308) — with the LES-realism quality-control gate re-verified solid + its production default locked (305). The pipeline stays structurally complete, exhaustively tested, CI-green. **Iters 310–319 (codex still capped Jun 24):** the differentiable path's masked reductions are now FULLY gradient-locked (316/317 + 299); the deployer's provenance is complete (health verdict 312 + two-sided window alignment 313); a multi-day SLURM run now shows real-time per-round progress + line-search activity (318/319, it was silent before); a clamping-safety tripwire (314) + the distributed/HPC path validated under real MPI (315) round it out — the in-repo proof is non-vacuous for single AND multi-coefficient recovery (to 1e-6). **Iters 337–350 (HYBRID-COORDINATE correctness thread — the biggest find since the launch bug):** the default `GridConfig.vertical_coord="hybrid"` (`p = A·p_ref + B·p_s`) was silently handled as pure-sigma (`p = σ·p_s`) throughout compare-reanalysis, so the whole pipeline was QUIETLY WRONG over terrain (`p_s ≠ p_ref`) — bias mass weights, reference regrid, LES-spin-off θ/heights/geopotential, env-kernel CAPE/shear, AND the LES large-scale subsidence ω (45.6% wrong at a 700-hPa column). ALL sites are now fixed to the coordinate's true `pressure_at_full/half` (byte-identical for pure-sigma) or, for ω, to a REUSE of the canonical hybrid dycore blocks (`compute_mass_flux_hybrid`+`compute_omega_hybrid`); verified MPI-safe, env-consistent, differentiable, and locked by terrain-specific tests + a full non-literal pure-sigma sweep confirming no in-scope site remains (compressed row below). The four production changes (340/341/342/348) await the mandatory codex review (cap → Jun 24); everything else is test/docstring-only. **Iters 391–400 (compressed row below):** operator-readiness pre-flights (regional-ERA5 silent-extrapolation coverage WARNING 391); TWO real go/no-go bugs found by adversarial BLOW-UP probing of the gates (the held-out `--baseline-restart` verify 392 + the campaign monotonic gate 393 each FALSE-passing a NaN-masked diverged run — per-column RMSE masks a blown-up state to a spurious ≈0 bias) fixed by guarding the raw STATE finite + deduped onto the pre-existing `assert_model_state_finite` (394); an EXHAUSTIVE AD-safety sweep CONFIRMING the differentiability invariant holds chain-wide (all 3 coeff diagnoses + reductions + metric + deploy gradient-tested through their invalid branches) and closing the lone gaps (env-kernel no-neighbour GRADIENT 396, a direct `test_scm_rce_metrics.py` locking the gradient-safe-sqrt training-convergence property 397, the CMIP off-grid surface-field guard 395); and HPC-reproducibility hardening (a bit-exact campaign RESUME-EQUIVALENCE integration lock 398 → a fail-loud `_assert_resume_config_carries_field` resume-contract guard on both single+multi paths 399, which also CAUGHT + fixed a self-inflicted 389/391 pre-flight regression). All mutation-proven. At **iter 400** the FULL end-to-end pipeline — including the slow real-dycore correction e2e + the real CMIP coupled-run compare — is verified GREEN. Production changes (391/392/393/394/399) await the mandatory codex review (cap → ~Jun 24).
./docs/COMPARE_REANALYSIS.md-5-**Date:** 2026-06-15 (design); 2026-06-17 (impl began)
./docs/COMPARE_REANALYSIS.md-6-**Scope:** Atmosphere component only. ERA5 reanalysis only. **Not** supervised learning.
./docs/COMPARE_REANALYSIS.md-7-
./docs/COMPARE_REANALYSIS.md-8-> ## Implementation status (compressed at iter 10; full history in git log `feat/compare-reanalysis`)
./docs/COMPARE_REANALYSIS.md-9->
--
./docs/COMPARE_REANALYSIS.md-43-> | **MULTI-DAY offline ERA5 REFERENCE (`--era5-n-days`) — the reference-side fix matching iter-459's forcing fix (iter 460)** — *closes the last weather-vs-climate gap in the offline comparison + the every-10 compression* | iter 459 let the offline SST FORCING span a multi-month window, exposing the matching gap on the REFERENCE side: the offline ERA5 reference was ONE day's chunk (`open_local_era5_dataset(date)`, ~24 hourly times), so a multi-day/-month model climatology was compared to a single-day ERA5 SNAPSHOT — weather-vs-climate. FIX: `open_local_era5_dataset_multiday(dir, start, n_days)` concatenates `n_days` CONSECUTIVE daily chunks along time (`_consecutive_days`, month/year/leap rollover via `datetime`; each day independently merged+aligned so a month-boundary day picks the right monthly sfc chunk; n_days=1 == the single-day open, byte-identical). New `--era5-n-days` flag (offline-only; Zarr already spans times) → main builds the multi-day dataset, with a FAIL-LOUD guard if `--era5-time-idx + --era5-n-times` exceeds the loaded `~24·n_days` coverage. Tests: multi-day concat (8 times, monotonic, all vars) + n_days=1 == single-day + missing-day fail loud + `_consecutive_days` rollover/leap (11 offline + 9 runbook green, lint clean). Runbook §3 documents pairing `--era5-n-days D --era5-n-times <=24·D`. So a multi-day model climatology now compares to a matched multi-day ERA5 climatology — both SST forcing AND reference span the window. | 460 |
./docs/COMPARE_REANALYSIS.md-44-> | **Offline realistic-AMIP COMPARISON-VALIDITY arc (iters 451–459)** — COMPRESSED at iter 460 (header narrates; git log has per-iter detail) | The decade that made the fully-offline realistic AMIP-vs-ERA5 comparison physically consistent + operator-runnable, all CODEX PENDING (cap → ~Jun 24): **OCEAN-only ranking** so the LES budget targets closure-attributable columns (`ocean_valid_mask` + the `static_land_fraction` probe, ranking AND the monotonic gate both ocean-masked, 451) + the held-out VERIFY made ocean-consistent (452) + a CMIP applicability caveat (457); **LAND for a valid comparison** — `--land-mask-path` exposed (454, the default flat config compared model-ocean to ERA5-land over continents — apples-to-oranges dominating the ranking), its ERA5-`lsm`/latlon chain locked end-to-end (455), and a land-mask AMIP run verified STABLE (the slab-land engages, state finite, 456); **SEASONAL alignment + SST consistency** — verified the iter-449 insolation seam reaches RRTMGP not just gray (457), found the single-month offline forcing CYCLICALLY REPEATS against an advancing insolation on a long run (`get_forcing_at_time` wraps a sub-annual forcing) and guarded it (458), then BUILT the fix: multi-month `--amip-forcing-n-months` concatenation (459); **HPC feasibility** — a persistent JAX compilation cache to amortize the rrtmgp JIT across the self-requeue (453, the within-job per-round recompile needs the traced-C_K refactor, tracked). Every change default-safe + directly tested; the offline path now has consistent SST + insolation + land + forcing-window. | 451–459 |
./docs/COMPARE_REANALYSIS.md-45-> | **Turnkey `--align-insolation`: auto-derive `insolation_start_doy` from the offline ERA5 date (iter 450)** — *one-flag operational completion of the iter-449 seam* | Made the iter-449 seasonal-insolation capability turnkey for the campaign operator. Added an OPT-IN `--align-insolation` flag (+ `ALIGN_INSOLATION` env var in the launcher sbatch): when set, `_maybe_align_insolation` computes the noleap day-of-year of `--local-era5-date` (`noleap_day_of_year`) and injects `insolation_start_doy` into the base config via `_replace` BEFORE the campaign builds drivers, so EVERY corrected round runs the matching solar season — no hand-editing the config JSON, no hand-computing the DOY. Fail-LOUD (`SystemExit`) when set without an offline date or with a malformed date (the season is undefined). OFF by default => production byte-identical (the iter-449 default). The off-season launch NOTE is suppressed when `--align-insolation` is on (the helper prints its own `insolation_start_doy=<doy>` confirmation instead). Runbook §1b now lists THREE alignment paths (the new flag; the manual config field; a January window). Test `test_align_insolation_sets_start_doy_from_offline_date` (off=unchanged, on+Sep1=>244 + `validate_strict` passes, on+missing/short/bad-month-13 date=>SystemExit, parser default OFF). 139 campaign+runbook+seam tests green, lint clean. The radiation-path behaviour (a non-zero offset) stays **CODEX PENDING** (cap → ~Jun 24). | 450 |
./docs/COMPARE_REANALYSIS.md-46-> | **Operator-readiness + AMIP-vs-ERA5 comparability arc (iters 438–449)** — COMPRESSED at iter 450 (header narrates; git log has per-iter detail) | A campaign of HPC operator-UX hardening + comparability-correctness work while codex stayed capped (→ ~Jun 24, all marked CODEX PENDING): **observability** — per-round bias TRAJECTORY surfaced in `CampaignSummary.report()`/JSON (438), de-duped vs the pre-existing top-level `biases` (439, a self-corrected missed pre-impl search) + a STALLED-campaign (all-rejected) plot regression lock (439); **deploy guard** — cross-grid env-kernel deploy AUTO-warns on a near-no-op low-coverage correction (440); **HPC entry-point hardening** — config-generator `validate_strict` before-write so a bad `--radiation` fails at generation not load (441); the C_K-sensitivity tool became a real GO/NO-GO pre-flight GATE (exit 0/1) + a `sys.path` bootstrap-bug fix + a `--radiation` flag (442), a turnkey compute-node `preflight_ck_sensitivity.sbatch` + a sbatch↔CLI flag-lock (444) after FINDING the realistic rrtmgp pre-flight is compute-node-only (443, rrtmgp graph-compile is the bottleneck, not data/coupling); **empirical gate validation on REAL ERA5 (gray)** — the gate exited 1 (NO-GO) on the idealization-dominated bias, REPRODUCING iter-412/417 with current code (combined bias 11.25, C_K moves it 0.010%, a 91 K free-trop T-bias at σ=0.096 dwarfs the total; the most C_K-controllable level is the BL, σ=0.97/0.43%) — concretely confirming the realistic rrtmgp config is essential for clause-6 (445); **comparability correctness** — the climatology time-mean now EXCLUDES the un-equilibrated spin-up (`run_to_column_mean(spinup_days=…)` → `--spinup-days` → `SPINUP_DAYS`, 446) and a spin-up-not-set launch WARNING (447); and the **insolation/ERA5 season** gap was found (447), pinned to JANUARY-based (448), and FIXED as an opt-in driver seam `insolation_start_doy`/`_insolation_day` across all SEVEN insolation sites (five `day_to_calendar` + two gray `daily_mean_insolation`), byte-identical at the default, validated by the bitwise restart test (449). All tests green + lint clean each iter; production behaviour unchanged (every behaviour-changing piece is default-off + CODEX PENDING). | 438–449 |
./docs/COMPARE_REANALYSIS.md-47-> | **RDA test single-sourcing + the END-TO-END verification sweep — every loop layer verified correct, 5 gaps closed, 1 real MPAS bug found (iters 427–437)** — COMPRESSED (the iter-438/439/440 rows + git log narrate per-iter detail) | **427–430 (test hygiene):** single-sourced the d633006 synthetic-RDA archive convention into `tests/_offline_era5_rda.py` (`write_forcing_archive`, `write_full_archive`, public `sfc_name`/`pl_name` — replacing 4 copy-pasted writers, no private cross-import), wired the HPC SLURM launcher + runbook §2b for the fully-offline realistic AMIP path (`LOCAL_ERA5_DIR`/`AMIP_FORCING`, exactly-one-source compare reference), validated the CMIP offline-compare path end-to-end through `main()`, and folded 417–426.  **431–437 (the verification sweep):** an exhaustive end-to-end audit of the WHOLE loop, every layer found correct + comprehensively tested — the iter-348 hybrid-ω fix (`omega_from_divergence`, verified CORRECT; caught a reviewer math error; softened a "byte-identical"→"analytically identical" docstring), the bias FOUNDATION (`compare_reanalysis` guards + `interp_pressure_to_sigma`), the ranking METRICS (`column_era5_metrics`), the HPC dycore continuity (the TILED/spectral path is consistent), the LES→C_K DIAGNOSIS, the feedback ACTUATOR, the acceptance GATE, the time-mean accumulator, the LES-budget clustering, the CMIP dispatch + column extraction, and the LES flux extractors (`resolved_turbulent_fluxes_plane`) — and every `training/*.py` has a direct test.  CLOSED 5 narrow real gaps (vector-wind grad-safety, hybrid-interp `p_full`, wp2 interior co-location, static-scatter array-bg accumulation, +1) and FOUND + characterized 1 real bug: the MPAS dycore mixes a flux-form `dp_s_dt` with an advective-`div` mass flux (a `v·∇dp` terrain inconsistency) — OFF the compare-reanalysis HPC critical path (which uses the spectral/tiled C-grid); a 1-line fix PREPARED via the new shared `compute_mass_flux_from_cumsum` helper (`grids/vertical.py`, which also dedups the C-grid inline closure), CODEX PENDING for the dynamical-core benchmark suite + the Jun-24 cap reset.  An 810-test health check is green; the SOLE remaining clause-6 gate is the HPC empirical demonstration. | 427–437 |
./docs/COMPARE_REANALYSIS.md:48:> | **OFFLINE realistic-AMIP path BUILT + validated end-to-end (iters 417–426)** — COMPRESSED at iter 430 (git log + rows 427–430 keep detail) | Sharpened the C_K-sensitivity characterization (417 per-level: C_K control is BL-LOCALIZED — **0.30%** at the BL vs **0.01%** at the strat top, ~30×, so the full-column insensitivity is the model's free-trop/RADIATION error, not the correction; 418 run-length: the BL fraction climbs 0.304%→0.474% over 1→6 days but a linear extrapolation needs **~130 days** to the 5% floor → the empirical demo needs BOTH equilibration AND realism — a precise HPC characterization, productionized as `ck_sensitivity_trend` + a `--days-sweep` classifier).  THEN built the OFFLINE realistic-AMIP path end-to-end: `build_era5_amip_forcing` (419 — the FORCING-side twin of `open_local_era5_dataset`; subsamples the monthly-hourly RDA sfc to daily to avoid OOM; verified vs real d633006 Sept-2017); PROVED it drives a real `ModelDriver` AMIP run via a +15 K SST differential (+6.8 K BL warming ⇒ consumed, not just loaded) + fixed a PRODUCTION silent SIC-units corruption (`(0-1)` not in the fraction allowlist → a wrong scale would ZERO the ice) (420); the TURNKEY campaign `--amip-forcing-from-local-era5` (421 — one archive supplies BOTH the forcing AND the compare; `apply_amip_forcing_to_config` field map, fixed a dropped-`T_ice`-override bug); the offline forcing through the WHOLE correction loop in AMIP mode (422); REALISTIC radiation (rrtmgp) + the forcing RUNS (423 — asserts rrtmgp actually ran via `held_dT_rad`); the dry-run pre-flight CONFIRMS the forcing build (424); the COMPLETE turnkey path through `main()` (425); and the clause-6 bias REDUCTION in AMIP mode (426 — the AMIP twin: apply the known C_K=1.0 → re-run = truth → bias→0, the closest in-repo proxy to the HPC demo).  Every step Codex-reviewed (manual; cap → Jun 24); each found + fixed real issues (consumption, SIC-units, T_ice, nanargmax, the run-length HAZARD).  CODEX PENDING (real review on cap reset): 419/420/421/424 production. | 417–426 |
./docs/COMPARE_REANALYSIS.md-49-> | **REAL-ERA5 arc: unblocked → first compares → C_K-sensitivity CHARACTERIZED (iters 408–416)** — COMPRESSED at iter 420 (git log + rows 417–420 keep detail) | Real ERA5 found LOCALLY accessible (NCAR-RDA `d633006`, `ll025` NetCDF; gcsfs/network down) → `load_era5_slice(ds=)` injection (408) + the `open_local_era5_dataset` archive adapter (409) feed real ERA5 through the SAME regrid chain OFFLINE.  Campaign made LAUNCHABLE offline (`--local-era5-dir`, exactly-one-source guard, 410) + the FIRST real-model-vs-real-ERA5 compare (411, finite bias 11.16).  KEY CHARACTERIZATION: the idealized 1-day aquaplanet/gray-rad model's real-ERA5 bias is C_K-INSENSITIVE — only **0.01%** C_K-controllable full-column (412) — so the monotonic loop correctly REJECTS every round (the empirical clause is HPC-gated, NOT a pipeline bug).  Clause 6 (params IMPROVE the bias) DEMONSTRATED with REAL model physics via a twin OSSE (413: truth=model@C_K=1.0, biased=@0.4 → diagnosed C_K → re-run → bias→0, gate ACCEPTS).  Real-ERA5 INPUT path adversarially VALIDATED + ambiguous-glob guard (414).  FULL loop end-to-end with real ERA5 + a REAL LES → gate correctly REJECTS the non-improving correction (415).  PER-LEVEL C_K (416): the BL is 23× more C_K-controllable than the free-trop top — C_K control is LOCALIZED to the boundary layer (VINDICATES the approach); the full-column insensitivity is the model's free-trop/radiation idealization, NOT the correction.  All codex-reviewed (manual; Codex capped); CODEX PENDING: 408/410/414 production. | 408–416 |
./docs/COMPARE_REANALYSIS.md-50-> | **Comprehensive manual adversarial-review sweep — every module + the glue validated, findings fixed (iters 401–407)** — COMPRESSED at iter 410 (header narrates; git log has per-iter detail) | Codex capped (→ Jun 24), so ran MANUAL adversarial reviews (the mandated step) across the WHOLE pipeline, fixing each finding + validating the rest, all mutation-proven.  REAL BUGS/GAPS FIXED: `model_state_is_finite` missed the MPAS `u_edge` prognostic (401 — a diverged edge-velocity state could pass the go/no-go); a ncol=1 scalar-config resume-guard false-negative (401); a `slice_override_columns` empty-rank crash (404); `EnvKernel` had no predictor-name provenance ⇒ a wrong-ORDER deploy silently miscorrelates (405).  VALIDATED CORRECT + subtle untested invariants LOCKED: the hybrid-coordinate numerics — mass-weights/regrid/ω all byte-identical to pure-sigma (402); the surface-flux LES forcing — exner T→θ + ρ-cancellation + warm-SST sign, CLI-reachable + fail-loud + the dry-run surfaces it (403); the OSSE clause-5/6 PROOF — `recovered` needs BOTH bias-fell AND param-recovery, `bias_only`/`diverged`/`out_of_hull` guard the false-positives, RMS-norm locked so a scattered-final isn't a 'recovery' (406); the orchestration GLUE — `model_ctx` is the MODEL run (the diagnosis's column source) NOT the reference (407); the distributed cross-rank top-k — partition-independent tie-break matching serial (407 confirm).  Production fixes ⇒ CODEX PENDING. | 401–407 |
./docs/COMPARE_REANALYSIS.md:51:> | **Operator-readiness + adversarial bug-hunt + AD/resume hardening (iters 391–399)** — COMPRESSED at iter 400 (header narrates; git log has per-iter detail) | Operator HPC-readiness: a regional-ERA5 silent-extrapolation coverage WARNING (391). TWO REAL production bugs found by adversarial BLOW-UP probing of the GO/NO-GO gates (the highest-leverage place in a mature pipeline): the held-out `--baseline-restart` verify FALSE-PASSED a blown-up correction (392) and the campaign monotonic gate FALSE-ACCEPTED one (393) — per-column RMSE NaN-MASKS a diverged state to a spurious ≈0 bias ⇒ wrong verdict; fixed by guarding the raw STATE finite (NOT the masked scores), mutation-proven, then DEDUPED onto the pre-existing `assert_model_state_finite` + corrected its stale docstring that had CAUSED 393 (394). An EXHAUSTIVE AD-safety sweep then CONFIRMED the differentiability invariant holds chain-wide (all 3 coeff diagnoses C_K/Pr_t/C_eps + reductions + metric + deploy gradient-tested through their invalid branches) and closed the lone gaps: the env-kernel no-neighbour GRADIENT (396) and a direct `test_scm_rce_metrics.py` locking the gradient-safe-sqrt training-convergence property (397); plus the CMIP off-grid surface-field guard (395). HPC reproducibility: a bit-exact campaign RESUME-EQUIVALENCE integration lock (398) that surfaced the resume contract → a fail-loud `_assert_resume_config_carries_field` guard on both single+multi paths (399), which ALSO caught + fixed a self-inflicted 389/391 regression (the era5 pre-flights preempting later CLI guards). All mutation-proven; production changes (391/392/393/394/399) ⇒ CODEX PENDING (cap → ~Jun 24). | 391–399 |
./docs/COMPARE_REANALYSIS.md-52-> | **Operator HPC-readiness: clear --era5-time-idx bounds error + docs compress 380–389 (iter 390)** — *production; CODEX PENDING* | Continuing the iter-389 operator-readiness vein: a typo'd `--era5-time-idx` / held-out index (the runbook directs the operator to pick one) raised xarray's generic 'index N is out of bounds for axis 0 with size M' deep in `load_era5_slice`'s `ds.isel(time=…)`.  Added a campaign-specific bounds check naming the store's ACTUAL time count; SAME `IndexError` type (callers/tests unaffected), negative indices kept exactly as xarray (valid `[-n, n-1]`), `time`-dim-absent ⇒ skip + let isel raise; covers BOTH the single + time-mean paths (`load_era5_time_mean` reuses `load_era5_slice`).  Added `test_load_era5_slice_out_of_range_time_idx_gives_clear_error` (0/−1 valid; 1/−2 raise).  MUTATION-PROVEN non-vacuous: removing the check ⇒ xarray's 'out of bounds' ≠ my 'out of range' regex ⇒ fail; production restored.  Green (14 era5-load tests); zero lint.  Did the every-10-iters docs compression (folded 380–389 below).  Production, operator-facing ⇒ **CODEX PENDING**. | 390 |
./docs/COMPARE_REANALYSIS.md-53-> | **EXHAUSTIVE pipeline review COMPLETED + operator HPC-readiness begun (iters 380–389)** — COMPRESSED at iter 390 (git log has per-iter detail) — *production change 389 (--era5-zarr pre-flight) awaits CODEX (cap)* | The adversarial SELF-review (begun 376) reached EVERY subsystem; EVERY production path found CORRECT, each integration gap closed with a MUTATION-PROVEN non-vacuous guard.  **Campaign pass-throughs (380–382):** completed — l_mix_max multi/C_eps (380), env-kernel CAPE/shear cross-path + phis threading (381), area_weights + env_scales (382) — coordinate/l_mix_max/phis/env_scales all guarded vs silent default-reversion.  **Deploy + env-kernel (383):** saved-JSON→fresh-kernel chain + grid fingerprint guard + cross-res apply all correct; locked the mixed partial-coverage per-column dispatch (some→regression, some→background) with a self-caught-vacuity fix (distinct non-zero background).  **Integrity + composition (384):** a 396-test cross-cutting sweep green; surface-flux × multi-coefficient composition locked.  **Visualization (385):** the corrected-field-map reshape ORIENTATION (row-major) locked (the CLAUDE.md transpose/F-order artifact class).  **ERA5 INPUT side (386–388):** all 5 regrid paths' geographic orientation at 'reproduces-cell-latitude' strength; the level-axis `argsort` vs `[::-1]`; the q→mixing-ratio conversion locked in ALL 4 carries (latlon/MPAS/cubed-sphere/spectral); the `phis` regrid into the carry; + a 5-test @slow strongest-integration verification (capstones + e2e, green).  **Operator HPC-readiness (389):** found the empirical clause genuinely blocked (no local ERA5, no gcsfs/network, needs HPC scale); FIXED the missing `--era5-zarr` fail-fast pre-flight (typo ⇒ clear launch error, not a cryptic post-build zarr crash; remote-URI-safe).  The recurring gap pattern throughout: an integration test exercising a pass-through/field only at its DEFAULT value.  Production changes 389 (+ all 337–371) remain CODEX-PENDING (cap → Jun 24). | 380–389 |
./docs/COMPARE_REANALYSIS.md-54-> | **Surface-flux completion + OSSE/capstone clause-5 proofs + adversarial SELF-review thread (iters 370–379)** — COMPRESSED at iter 380 (git log has per-iter detail) — *production change 371 (MPAS surface flux) awaits CODEX (cap)* | THREE threads.  **(A) Surface-flux LES completion (370–372):** verified on cubed-sphere (370); COMPLETED on the 4th grid MPAS by REUSING `reconstruct_cell_velocity` (Perot edge→cell, differentiable, no new physics — superseding the iter-369 fail-loud) so it works on all 4 grids + is realism-gate-compatible (BL warming ~0.3 K << 3 K `theta_drift`) (371); locked through the REAL compressible-Euler LES (the helper's flux WARMS the column vs no-flux, not a mock) (372).  **(B) OSSE / capstone proofs of clause 5 (373–375):** a synthetic-run_fn OSSE proves the loop LOGIC REDUCES a known bias — the SIGN the real-model capstone can't assert — with a non-vacuity companion (373); extended to the HPC clustering cost path (`les_budget=K` → 1/4 the LES runs, bias still zeroed via a correct member→rep broadcast) (374); an AMIP real-rerun capstone closing the mode-coverage asymmetry (the loop closes for BOTH modes the done-criterion names, not just CMIP) (375).  **(C) Adversarial SELF-review thread (376–379, codex capped until ~06-24):** having established the loop is EXHAUSTIVELY covered (the production `perfect_model_osse.py` harness already proves recover/no-change/reject + cross-resolution transfer), manually reviewed the pending DEFAULT-path production physics + locked each finding with a mutation-proven non-vacuous test — the surface-flux helper (arg-order/sign/exner/ρ-cancellation all correct; independent-analytic lock, 376); the hybrid-coordinate COMPARE fix (full wiring correct; campaign-level `coordinate` pass-through guard, 377); the hybrid LES-forcing ω (reuses the dycore blocks with the same `dp_s_dt=-D_total_p/B_range`; cross-branch A=0 reduction lock, 378); the C_K/C_eps DIAGNOSIS (the exact inverse using the SAME `mixing_length`; tuned-`l_mix_max` threading guard, 379).  EVERY production review found NO bug; the recurring gap was a VACUITY TRAP — an integration test exercising a pass-through only at its DEFAULT value, so a hardcoded regression would pass — each closed with a tuned/non-default mutation-proven guard.  Production changes 371 (+ all prior 337–369) remain CODEX-PENDING (cap). | 370–379 |
./docs/COMPARE_REANALYSIS.md-55-> | **SURFACE-FLUX LES feature + final pipeline audit (iters 360–369)** — COMPRESSED at iter 370 (git log has per-iter detail) — *production changes 364–369 await CODEX (cap)* | TWO threads.  **(A) Final audit (360–362):** the clause-4 cluster BROADCAST is value-correct (distinct rep values land on the right members; validity counted over LES RUNS, not duplicated columns, 360); the CMIP coupled-SST env tag (`get_sst_sic` → the o2a-remapped ATM-grid SST, NOT raw ocean `T_sfc`) is correct + non-vacuously cross-grid-tested (361) ⇒ EVERY pipeline stage is now verify-first audited correct+tested; the runbook documents the iter-357/358 coordinate guard for the operator (362).  **(B) Surface-flux LES enhancement (363–369, default-OFF opt-in):** gives a worst-column spin-off LES a prescribed surface-flux BC so surface-driven/convective columns develop turbulence (raising the diagnosis-validity rate).  Scoped + initially DEFERRED as codex-blocked physics (363), then found the REUSE-based path (no new tunables): `column_surface_kinematic_fluxes` REUSES the GCM bulk scheme (`compute_surface_fluxes`) + converts to the kinematic θ/q_v fluxes the iter-151 BC consumes (`w'θ'=shflx/(ρ·c_pd)·exner`, `w'q'=lhflx/(ρ·L_v)`); differentiable + reuse-equivalent to rtol 1e-12 (364).  Wired END-TO-END: `ColumnLESConfig.surface_flux` flag → `process_column` (fail-loud w/o SST) → `make_les_diagnose_fn` threads `model_ctx.sst_K` (365) → `--surface-flux` CLI flag (366) → regression-verified + assumptions scoped for codex (ocean-`q_sfc`, C_K forcing-invariance, constant flux) (367) → output records `les_config` provenance (368).  SELF-SCRUTINY caught a REAL bug (369): `--surface-flux` + MPAS crashed (`jnp.asarray(None)` on `v=None`); fixed with fail-loud guards at TWO sites (`extract_gcm_column` backstop + `_build_run_setup` dry-run rejection); supported on the 3 cell grids.  The feature is COMPLETE, opt-in, bounded-risk (monotonic + realism gates), awaiting codex. | 360–369 |
./docs/COMPARE_REANALYSIS.md-56-> | **VERIFICATION & HARDENING decade (iters 350–359)** — COMPRESSED at iter 360 (git log has per-iter detail) — *production changes 351/352/357/358 await CODEX (cap)* | After the hybrid thread, a comprehensive verify-first audit + targeted hardening of the WHOLE pipeline.  **Diagnosis layer locked against the REAL forward** (the systematic-bias guard): C_K (353), Pr_t (354), C_eps (355) inverse diagnoses each round-trip-verified against the ACTUAL `clubb_lite_turbulence` — NOT hand-written replicas that share the formula — every lock MUTATION-VERIFIED (break the forward ⇒ test fails); C_eps used a clean S²/N²-cancelling dissipation-rate extraction (no duplicated numerics).  **Operator PRE-FLIGHT guards** (catch a misconfig in the dry-run, not after a multi-day run): per-column `area_weights`/`valid_mask` broadcastability (351) + `lat_deg`/`lon_deg` gather-shape (352) vs the reference grid; verify-CLI `--vertical-coord` (357) + hybrid `--p-top-Pa` (358) vs the restart's recorded run config — all provably-safe / defensive (no false positives).  **Dispatch-hardening** (350): `omega_from_divergence` raises on a non-σ/non-hybrid coordinate.  **Audited correct + (where needed) lock-strengthened**: resolved eddy fluxes `⟨w'φ'⟩`, C_K/c_eps level co-locations, the OSSE (mock-diagnosis appropriateness + a NEW harmful-diagnosis gate-safety demonstration 356), the time-mean accumulator (float64 precision), the env-kernel NW deploy + column clustering (359).  **@slow REAL-model+REAL-LES e2e + campaign one-round** re-run green (358) — the whole real chain still composes.  CONCLUSION: the ENTIRE pipeline is verify-first audited correct + tested; the hybrid thread was the one real bug, everything else correct with small gaps filled.  Disciplines: mutation-test every lock, vacuity-check, no-duplicate-numerics, defensive-guard-no-false-positives. | 350–359 |
--
./docs/performance/scaling/SCALING_STATUS_AUDIT.md-23-
./docs/performance/scaling/SCALING_STATUS_AUDIT.md-24-| grid | CPU-MPI | GPU / SPMD | true weak/strong evidence (a) | known blockers | next measurement |
./docs/performance/scaling/SCALING_STATUS_AUDIT.md-25-|------|---------|------------|-------------------------------|----------------|------------------|
./docs/performance/scaling/SCALING_STATUS_AUDIT.md-26-| **spectral** | (d) single-rank only (global Legendre transforms; no MPI path) | (d) N/A — both multi-device schemes measured anti-scaling (`spectral_level_shard_cliff.md`); fp64-only | none possible | O(N³) global transform | none — stays N/A unless a GPU-native SHT effort (SHTns/sphericart) is explicitly launched |
./docs/performance/scaling/SCALING_STATUS_AUDIT.md-27-| **cubed-sphere** | (a) genuine ≤6-face decomposition via `run_levante_gpu_scaling.py --cs-mpi-scatter` (bit-equal 1e-15 vs serial); default (no flag) is replicated dynamics, refused for scaling claims. Gloo/TCP multinode anti-scales (fabric, not code — `multinode_clean`) | (a)≤6 devices: face-sharded SPMD (`--cs-spmd`, single-process or multi-controller NCCL); Derecho/Levante NCCL job lanes exist (`scripts/cluster/scaling_*`) | 2-GPU strong 0.73–0.83 eff (Ginsburg, AT the PCIe roofline of that host) | >6 GPUs: (c) sub-face tiled production lanes wired (`_run_tiled_cube_spmd` blocked loop, per-segment `lax.scan` since M3b inc-1 — see lever 7) but NO >6-GPU hardware receipts; envelope = default-config dycore + Kessler / operator-split unified physics | production-size Derecho/Levante multi-node NCCL runs incl. the >6-GPU tiled lanes |
./docs/performance/scaling/SCALING_STATUS_AUDIT.md:28:| **icosahedral / MPAS** | (a) graph-partition domain decomposition (METIS/RCB/SFC `auto`), validated vs serial ~1e-9 | (a) multi-GPU via route-A mpi4jax (opaque to XLA overlap); (c) route-B multi-controller NCCL SPMD wired since M3c #981 (native ppermute step, `bench_mpas_spmd_scaling --multicontroller`, cluster lane E; XLA collective-permute combining defaulted for the lane, #1113) — no production-scale receipts yet | Derecho 2026-07-02: CPU-MPI to 128 ranks f32 ~75× (eff ~0.59); GPU 1→16 A100 @28 km ~6× (eff ~0.38), coarse grids flat (per-device floor, not a defect) | route-A mpi4jax leg is latency-bound, no comm/compute overlap; route-B ppermute round count (edge-coloring rounds × launch latency, #1113) | production-size lane-E runs; re-measure 8→16 GPU leg on native ppermute |
./docs/performance/scaling/SCALING_STATUS_AUDIT.md-29-| **lat-lon** | (a) latitude-band decomposition (`make_latlon_mpi_step`, #641; pole_bc='wall'), validated; driven by `run_levante_gpu_scaling.py --grid latlon` | (a) lat-band SPMD `bench_atm_latlon_spmd_scaling.py`: single-process multi-device + `--multicontroller` route-B (native NCCL ppermute, no mpi4jax) | Derecho: CPU-MPI 128 ranks f32 ~30× (eff ~0.23 — 1-D band perimeter cost, as designed); GPU 16 A100 @28 km 7.5× (eff ~0.47, route-A). Ginsburg 2-GPU: strong 1.10×, weak eff 0.64 (PCIe-capped) | 1-D band decomposition perimeter at high rank counts; 2-D latlon decomposition untested on a real fabric | production-size native-NCCL multicontroller runs on Derecho/Levante (job lanes C exist) |
./docs/performance/scaling/SCALING_STATUS_AUDIT.md-30-
./docs/performance/scaling/SCALING_STATUS_AUDIT.md-31-## Ocean support matrix
./docs/performance/scaling/SCALING_STATUS_AUDIT.md-32-
./docs/performance/scaling/SCALING_STATUS_AUDIT.md-33-| grid | CPU-MPI | GPU / SPMD | true weak/strong evidence (a) | known blockers | next measurement |
--
./docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-12-  agent's "OFF by default" is the XLA default; our backend overrides it.
./docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-13-  Explains the campaign note "XLA already overlaps GPU collectives".
./docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-14-- **XLA command-buffer / CUDA-graph capture** (#4): ALREADY set
./docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-15-  `xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL,COLLECTIVES` (line 199)
./docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-16-  — CUSTOM_CALL is included, so the LAPACK gtsv FFI IS in the captured set
./docs/performance/scaling/literature_scan_2026-06-13_new_levers.md:17:  (no graph fragmentation from it). (Worth a one-off Nsight trace to CONFIRM
./docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-18-  the FV3 step is one graph, but the flag coverage is already correct.)
./docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-19-- **Pipelined/communication-avoiding Krylov** (#5): equivalent SHIPPED
./docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-20-  (Chronopoulos-Gear single_reduce + Chebyshev reduction-free precond). The
./docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-21-  2025 POP/CG-variant papers beat a naive PCG baseline we're already past.
./docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-22-- **Ocean f32 path** (part of #2): ALREADY clean + measured 1.72x LL128
--
./docs/performance/scaling/RESUME_STATE_2026-06-13.md-21-| f0c27da1 | cadence-less runs no longer collapse to 1-step segments (serial 5×) |
./docs/performance/scaling/RESUME_STATE_2026-06-13.md-22-| 65b29082 | atm F1: pack T+ln(ps) SPMD halo into one collective |
./docs/performance/scaling/RESUME_STATE_2026-06-13.md-23-| a3b9c597 | ocean zonal_line preconditioner (cyclic-Thomas, comm-free) |
./docs/performance/scaling/RESUME_STATE_2026-06-13.md-24-| 1f142625 | ocean vertex-mask hoist out of the traced step |
./docs/performance/scaling/RESUME_STATE_2026-06-13.md-25-| 1a0099a0 | barotropic levers documented OPT-IN (regime crossover) |
./docs/performance/scaling/RESUME_STATE_2026-06-13.md:26:| e39b027d | distance-to-limit assessment |
./docs/performance/scaling/RESUME_STATE_2026-06-13.md-27-| 7a7100d6 | neumann-fill field+mask halo packed via dtype cast (census 38→32) |
./docs/performance/scaling/RESUME_STATE_2026-06-13.md-28-| 36861ecb | lat-lon multi-GPU state shard_state grid_type fix + loud tiled-halo skip (codex P1) |
./docs/performance/scaling/RESUME_STATE_2026-06-13.md-29-| 512ab62b | parallelization literature review |
./docs/performance/scaling/RESUME_STATE_2026-06-13.md-30-
./docs/performance/scaling/RESUME_STATE_2026-06-13.md-31-## Codex review (2026-06-13) — all P0/P1 resolved
--
./docs/performance/scaling/scaling_indicators.csv-19-2026-06-13,0293f865,chebyshev,ocean_latlon,mpi,precond,2.31,speedup_iters_d8,8478404,cheby deg8 35 iters to rel1e-6 vs jacobi >80 (budget-censored lower bound); coastal 48x96; cuts latency-bound global reductions >=2.3x; deg4=72iters
./docs/performance/scaling/scaling_indicators.csv-20-2026-06-13,7d153934,tridiag_lapack,atm_cpu,cpu,vmix,13.1,tridiag_lapack_cpu,8478438,LAPACK gtsv vs fori-loop Thomas CPU; C96-scale 55296col x72lev 4143ms->317ms; 8.4x@20k/32 13.0x@50k/64; parity 8.9e-16; opt-in LEGOESM_TRIDIAG=lapack
./docs/performance/scaling/scaling_indicators.csv-21-2026-06-14,ec1b0a36,lnps_pack,atm_cube,spmd,halo,2,collective_cut,8481289,PE ln_ps+hybrid_factor ride the {zeta,B,1/T} stage pack; -2 cross-rank collectives/RK3-substep (hybrid AMIP) on multinode-SPMD/CPU-MPI; provable-by-construction count-cut; SPMD16/16+MPI5/5 parity (bit-identical); re-audit lever
./docs/performance/scaling/scaling_indicators.csv-22-2026-06-14,83a180bd,divv_pack,atm_cube,spmd,halo,3,collective_cut,8481480,div_v completes the PE stage-pack; ALL 6 cube-PE scalars {zeta,B,1/T,ln_ps,hf,div_v} in ONE collective; -3 cross-rank collectives/RK3-substep (div_damp+hybrid AMIP, non-async); SPMD16/16+MPI5/5 parity (bit-identical)
./docs/performance/scaling/scaling_indicators.csv-23-2026-06-11,e209bbb0,a1_spmd,atm_cube,spmd,capability,6,tiled_np_validated,8460192,face-SPMD cube ceiling = 6 faces (1 device/face) before sub-face tiling
./docs/performance/scaling/scaling_indicators.csv:24:2026-06-14,7d5c5188,tiled_np24,atm_cube,spmd,capability,24,tiled_np_validated,8482583,full fv3_sw_tendencies sub-face tiled np24=6x2x2 bit-EXACT on 2 nodes (200 cross-proc collective-permute) — np54 host-gate; np>6 future-HW (anti-scales Gloo-TCP so capability not speedup)
./docs/performance/scaling/scaling_indicators.csv-25-2026-06-14,eb1fe737,tiled_3d_vort,atm_cube,spmd,capability,1,ops3d_np24,8484502,dgrid_vorticity = first fv3_hydrostatic_tendencies (3D PE) op np24-tiled 4D bit-identity (host + shard_map); grows as cc2c/geopotential/ln_ps PGF tile toward the full 3D stage
./docs/performance/scaling/scaling_indicators.csv-26-2026-06-14,2fc8897c,tiled_3d_alinterp,atm_cube,spmd,capability,3,ops3d_np24,8485473,arakawa_lamb_gradient + interp_corner_to_center np24 4D stages added (now 3 shared 3D-PE cc-ops np24-tiled: +dgrid_vorticity); bit-identity host+shard_map
./docs/performance/scaling/scaling_indicators.csv-27-2026-06-14,8fd4ef0d,tiled_3d_vel,atm_cube,spmd,capability,5,ops3d_np24,8485629,dgrid_to_cgrid + dgrid_to_center_vector np24 4D (both LOCAL within-face; no vector halo unlike SW) — now 5 shared 3D-PE cube ops np24-tiled
./docs/performance/scaling/scaling_indicators.csv-28-2026-06-14,58475e41,tiled_3d_geo,atm_cube,spmd,capability,6,ops3d_np24,8485746,compute_geopotential (Simmons-Burridge vertical integration) np24-tiled — vertical-local per-column exact-partition; 6 shared 3D-PE ops np24-tiled
./docs/performance/scaling/scaling_indicators.csv-29-2026-06-14,2c53abea,ocean_spmd_pcg,ocean_latlon,spmd,strong,0.3087,eff_2gpu_ocean_pcg_standard,8486096,FIRST ocean-GPU SPMD number — latlon 360x720 M60 f64 barotropic PCG on 2x RTX8000 PCIe (1GPU 5.06ms 2GPU 8.20ms) kernel anti-scales (psum-latency-bound — audit wall MEASURED)
./docs/performance/scaling/scaling_indicators.csv-30-2026-06-14,2c53abea,ocean_spmd_pcg,ocean_latlon,spmd,strong,0.4280,eff_2gpu_ocean_pcg_single_reduce,8486096,single-reduce (1 psum/iter lever #1) cuts 2-GPU penalty to 7.59 vs 8.20ms — opt-in for the comm-bound multi-GPU regime (1-GPU slower: extra matvec)
./docs/performance/scaling/scaling_indicators.csv:31:2026-06-14,246df03f,ocean_fullstep_2gpu,ocean_latlon,mpi,strong,0.659,eff_2gpu_ocean_fullstep,8486172,FULL ocean step 2xRTX8000 mpi4jax HOST-STAGED 180x360 nlev30 implicit_cn f64 (1GPU 27.0ms 2GPU 20.5ms 1.32x) — AMORTIZATION CONFIRMED barotropic-only anti-scales 0.31 to full-step 0.66 (cuda-aware MPI next lever toward 0.8-0.93)
./docs/performance/scaling/scaling_indicators.csv-32-2026-06-14,bf9c21f4,ocean_fullstep_def,ocean_latlon,mpi,strong,0.624,eff_2gpu_ocean_fullstep_180n30,8486182,SOLVER-MATCHED (np1 force-pcg) full ocean step 2xRTX8000 180x360 nlev30 (np1 25.5ms np2 20.4ms 1.25x) — small config comm-bound; GPU-distinctness PROVEN CVD0/CVD1 distinct PIDs
./docs/performance/scaling/scaling_indicators.csv-33-2026-06-14,bf9c21f4,ocean_fullstep_def,ocean_latlon,mpi,strong,0.920,eff_2gpu_ocean_fullstep_360n60,8486182,SOLVER-MATCHED full ocean step 2xRTX8000 360x720 nlev60 (np1 272ms np2 148ms 1.84x) NEAR-IDEAL at production scale — amortization eff rises 0.62 to 0.92 with problem size; host-staged MPI = lower bound (cuda-aware next)
./docs/performance/scaling/scaling_indicators.csv-34-2026-06-14,32a1d490,atm_latlon_2gpu,atm_latlon,mpi,strong,0.821,eff_2gpu_atm_latlon,8486204,FIRST atm lat-lon 2-GPU (route-A overlay venv mpi4jax host-staged) 180x360 nlev26 held_suarez f64 (np1 18.9ms np2 11.5ms 1.64x) — explicit FV PE dycore halo-bound scales cleanly (no ocean barotropic bottleneck); eff rises 0.72 res90 to 0.82 res180
./docs/performance/scaling/scaling_indicators.csv-35-2026-06-14,046b7195,ocean_weak_2gpu,ocean_latlon,mpi,weak,0.570,eff_2gpu_ocean_weak,8486221,OCEAN WEAK 2-GPU (route-A host-staged rows/rank=90 nlev60) np1 11.3ms np2 19.9ms = weak eff 0.57 (nlev30=0.46) — LOWER than strong (barotropic 120-allreduce/step latency is the weak Amdahl term that does NOT shrink); eff rises with nlev like strong
./docs/performance/scaling/scaling_indicators.csv-36-2026-06-14,d0f9e50f,atm_ico_2gpu,atm_icosahedral,mpi,strong,0.829,eff_2gpu_atm_icosahedral,8486333,atm ICOSAHEDRAL (Voronoi METIS cell-partition) 2-GPU route-A I6 1.06M cells nlev26 held_suarez f64 (np1 30.5ms np2 18.4ms 1.66x); I5 eff 0.66 — completes atm all-grids 2-GPU (latlon 0.82 cube 0.73 ico 0.83; spectral=by-design cliff); halo-bound eff rises with size
--
./docs/performance/scaling/spmd_message_census_2026-07-08.md-24-per-row as `hlo_collective_permutes`:
./docs/performance/scaling/spmd_message_census_2026-07-08.md-25-
./docs/performance/scaling/spmd_message_census_2026-07-08.md-26-```python
./docs/performance/scaling/spmd_message_census_2026-07-08.md-27-# XLA_FLAGS=--xla_force_host_platform_device_count=N  JAX_PLATFORMS=cpu
./docs/performance/scaling/spmd_message_census_2026-07-08.md-28-hlo = jit_step.lower(state, dt).compile().as_text()
./docs/performance/scaling/spmd_message_census_2026-07-08.md:29:n_cp = hlo.count("collective-permute-start") + hlo.count("collective-permute(")
./docs/performance/scaling/spmd_message_census_2026-07-08.md-30-```
./docs/performance/scaling/spmd_message_census_2026-07-08.md-31-
./docs/performance/scaling/spmd_message_census_2026-07-08.md-32-Counted alongside `all-gather` / `all-reduce` occurrences.
./docs/performance/scaling/spmd_message_census_2026-07-08.md-33-
./docs/performance/scaling/spmd_message_census_2026-07-08.md-34-## Results
./docs/performance/scaling/spmd_message_census_2026-07-08.md-35-
./docs/performance/scaling/spmd_message_census_2026-07-08.md-36-### Cube cs-spmd full PE step (C24/L8 shape; counts are shape-independent)
./docs/performance/scaling/spmd_message_census_2026-07-08.md-37-
./docs/performance/scaling/spmd_message_census_2026-07-08.md:38:| devices | collective-permutes / step | all-reduce / step |
./docs/performance/scaling/spmd_message_census_2026-07-08.md-39-|---|---|---|
./docs/performance/scaling/spmd_message_census_2026-07-08.md-40-| 2 | 15 | 1 |
./docs/performance/scaling/spmd_message_census_2026-07-08.md-41-| 3 | 26 | 1 |
./docs/performance/scaling/spmd_message_census_2026-07-08.md-42-| 6 | 46 | 1 |
./docs/performance/scaling/spmd_message_census_2026-07-08.md-43-
--
./docs/performance/scaling/spmd_message_census_2026-07-08.md-61-coarse cube curves, and the 2-GPU dip (15 CPs but 3 faces/shard + sync per
./docs/performance/scaling/spmd_message_census_2026-07-08.md-62-CP on a halved compute slice).
./docs/performance/scaling/spmd_message_census_2026-07-08.md-63-
./docs/performance/scaling/spmd_message_census_2026-07-08.md-64-### Atm latlon SPMD step (64×128/L8, 4 devices)
./docs/performance/scaling/spmd_message_census_2026-07-08.md-65-
./docs/performance/scaling/spmd_message_census_2026-07-08.md:66:| arm | collective-permutes / step |
./docs/performance/scaling/spmd_message_census_2026-07-08.md-67-|---|---|
./docs/performance/scaling/spmd_message_census_2026-07-08.md-68-| baseline | 41 |
./docs/performance/scaling/spmd_message_census_2026-07-08.md-69-| `LEGOESM_LATLON_SPMD_FUSED_HALO=1` | **29 (−29%)** |
./docs/performance/scaling/spmd_message_census_2026-07-08.md-70-
./docs/performance/scaling/spmd_message_census_2026-07-08.md-71-The fused multi-pad (audit item 7, shipped opt-in for the ocean with a
--
./tests/atmosphere/hydrostatic/unit/test_topography.py-394-    """T6: geopotential-vs-meters double-g trap.
./tests/atmosphere/hydrostatic/unit/test_topography.py-395-
./tests/atmosphere/hydrostatic/unit/test_topography.py-396-    An ERA5-style invariant file where 'z' is GEOPOTENTIAL [m**2 s**-2] used
./tests/atmosphere/hydrostatic/unit/test_topography.py-397-    to be treated as elevation [m] and multiplied by g again in
./tests/atmosphere/hydrostatic/unit/test_topography.py-398-    load_real_topography — phis silently inflated ~9.8x.  The loader must
./tests/atmosphere/hydrostatic/unit/test_topography.py:399:    reject implausible magnitudes (> _MAX_PLAUSIBLE_ELEV_M) and geopotential
./tests/atmosphere/hydrostatic/unit/test_topography.py-400-    units attributes.
./tests/atmosphere/hydrostatic/unit/test_topography.py-401-    """
./tests/atmosphere/hydrostatic/unit/test_topography.py-402-
./tests/atmosphere/hydrostatic/unit/test_topography.py-403-    def setUp(self):
./tests/atmosphere/hydrostatic/unit/test_topography.py-404-        self.tmpdir = tempfile.mkdtemp()
--
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-6-kept that signal off the actual Derecho/Levante runs. Read the campaign context
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-7-first: `derecho_levante_sota_review_2026-07.md` (measured baselines + SOTA),
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-8-`spmd_message_census_2026-07-08.md` (the message-count analysis this extends),
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-9-`SCALING_STATUS_AUDIT.md` (support matrix).
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-10-
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:11:Every claim below is labelled **CONFIRMED** (measured / read from code) or
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:12:**PLAUSIBLE** (inferred, needs a machine receipt).
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-13-
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-14----
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-15-
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-16-## 1. The bottleneck table (what limits each axis, and why)
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-17-
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-18-| Component × axis | Limiter | Evidence | Tier |
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-19-|---|---|---|---|
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:20:| Atm cube strong (≤6 GPU) | **collective-permute COUNT** grows with shard count while per-msg NCCL p2p latency (~30–80 µs) doesn't amortise on small tiles → anti-scales | census: 12 CP @2dev, 41 @6dev (C24/L8, below); f64≡f32 curves ⇒ latency-bound | CONFIRMED |
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:21:| Atm latlon strong | 4→8 GPU node-crossing plateau on route-A (mpi4jax sendrecv OPAQUE to XLA latency-hiding scheduler ⇒ no comm/compute overlap) | Derecho 78 km throughput 600→480→recover@16; route-B multicontroller SHIPPED to remove it | CONFIRMED |
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:22:| Atm ico strong | best-scaling grid (low perimeter/area cell partition); multihost SPMD still open | Derecho 28 km ico eff ~0.38 @16 A100, still rising | CONFIRMED |
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:23:| Atm/all coarse | per-device saturation floor (<~30k cols/GPU flat/anti) — NOT a defect | 111 km latlon/ico FLAT 1→16 A100 | CONFIRMED |
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:24:| Ocean strong (CPU-MPI) | **implicit-CN PCG reduction wall**: default `pcg_variant="standard"` = 2 *sequentially-dependent* all-reduces/iter × `fixed_iters=60` ⇒ ~120 latency-serialized all-reduces/step | `barotropic_common.py` `_fixed_iteration_pcg` (p·Ap @:619 → dependent r·z @:626); np16→32 eff 0.55 | CONFIRMED |
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:25:| Ocean strong (mitigation) | `single_reduce` (Chronopoulos–Gear) → 1 all-reduce/iter; split-explicit + `barotropic_local_subcycle_clamp` → 3 all-reduce/step (reduction-free subcycle) | both exist, both NON-default; beats standard at ≥2 nodes 1.15–1.65× | CONFIRMED |
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:26:| Ocean weak | ~1.0 eff (as SOTA); the strong ceiling is naive land imbalance, not comm | 0.97 weak @590k cells/rank | CONFIRMED |
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:27:| Spectral | single-device by design (both multi-device schemes measured anti-scaling) | prior campaign | CONFIRMED |
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-28-
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:29:Two structural hot-loop items found in the code map (both PLAUSIBLE as
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-30-scaling costs at high step counts, neither a hot-loop collective):
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-31-- **Per-step host dispatch** in the operator-split SPMD driver
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-32-  (`model_driver.py:7783` Python `for` over `seg_steps`) — a compiled
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-33-  `lax.scan`-per-segment path exists (M3b) but the operator-split loop dispatches
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-34-  per step; host-dispatch overhead scales with step count.
--
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-45-GPU executes.
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-46-
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-47-Fresh census (sharded cube full PE step, `run_scaling_diagnosis.py --mode
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-48-census`, C24/L8, CPU virtual devices):
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-49-
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:50:| devices | collective-permute / step | all-reduce / step |
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-51-|---|---|---|
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-52-| 2 | 12 | 1 |
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-53-| 6 | 41 | 1 |
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-54-
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:55:Readings (CONFIRMED): the single all-reduce is the conservation fixer (fine).
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-56-The permute count scales ~linearly with the shard count (edge-coloring: each
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-57-face has 4 neighbours ⇒ 4 ppermute rounds × the exchange points) — this IS the
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-58-cube anti-scaling. (The absolute counts are lower than the 15/46 in the
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-59-2026-07-08 note — halo packing improved since — which is exactly why the count
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-60-now belongs on every row as a REGRESSION-tracked metric, not a scratch probe.)
--
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-62-## 3. Instrument gaps closed this session
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-63-
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-64-The diagnosis was well-characterised in the docs, but two gaps kept the signal
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-65-off the real machine runs:
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-66-
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:67:1. **The message census counted only collective-permutes.** The ocean PCG
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-68-   all-reduce wall — the #1 ocean strong-scaling limiter — was invisible to a
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-69-   permute-only census. Fixed: canonical `metadata.count_collectives()` /
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-70-   `hlo_collective_census()` count **all** families (permute + all-reduce +
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-71-   all-gather + all-to-all + reduce-scatter) with the same op-call-form
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-72-   discipline (config-header flag echoes never inflate; async `-start` counted
--
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-96-   selectable): re-run `bench_ocean_mpi_scaling.py` / the SPMD ocean bench with
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-97-   `--pcg-variant single_reduce` and with split-explicit +
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-98-   `barotropic_local_subcycle_clamp` at ≥16 ranks — the calculus flips toward
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-99-   the reduction-free path where per-message latency dominates.
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-100-3. **Lane-T GPU-runtime A/B** (already wired, `RUN_TUNE=1`): PGLE was the
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:101:   measured winner (+8.5 %); the XLA collective-permute-combine + pipelined-p2p
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-102-   arms still need SPLITTING to attribute (the combined arm hurt −10 %).
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-103-4. Per §1, cube >6 GPU needs the sub-face tiled production step (separate
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-104-   project); the count floor is otherwise reached in code.
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-105-
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-106-## 5. Verification
--
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-111-- `run_scaling_diagnosis.py --mode census` verified locally (2/6 virtual
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-112-  devices, table §2); the census phase records into `summary.json`.
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-113-- cube/mpas bench census gates green (`test_bench_cube_tiled_step_scaling`,
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-114-  `test_bench_mpas_spmd_gates`).
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-115-- `diagnosis.pbs` / `.sbatch`: `bash -n` clean; no hardware receipt yet
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:116:  (submit-ready, PLAUSIBLE until a real run lands).
--
./docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-191-- MPAS implicit_cn np>1: LOUD NotImplementedError guard (was silently
./docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-192-  rank-local stock CG + rank-local mass projection).  Distributed
./docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-193-  MPAS PCG remains the documented TODO (Voronoi halo A_op +
./docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-194-  owned-cell-masked dots).
./docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-195-
./docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md:196:Per-grid distance-to-limit verdicts:
./docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-197-- cubed-sphere atm GPU: per-device at JAX-peer limit; 2-GPU strong at
./docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-198-  PCIe link roofline (0.73 f64).  CPU-MPI dynamics still REPLICATED
./docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-199-  (A1) — the one architectural gap; needs jax.distributed SPMD face
./docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-200-  mesh (multi-week, decision item).
./docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-201-- lat-lon atm+ocean CPU-MPI: comm engineered down 63->~30 (ocean) and
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-100-OPEN ANOMALY, characterised not explained: per-GPU throughput dips at
./docs/performance/scaling/levante_campaign_2026-07-24.md-101-np=4 (247 Mc/s/GPU vs 430 at np2 and 306 at np8), so np4 is barely faster
./docs/performance/scaling/levante_campaign_2026-07-24.md-102-than np2 while np8 is 2.5x faster than np4. Evidence gathered:
./docs/performance/scaling/levante_campaign_2026-07-24.md-103-- REPRODUCIBLE: two repeats per arm agree within 1 % (19.92/19.87,
./docs/performance/scaling/levante_campaign_2026-07-24.md-104-  17.14/17.04, 6.87/6.97).
./docs/performance/scaling/levante_campaign_2026-07-24.md:105:- PLACEMENT REFUTED: np4 packed on one node (17.09 ms) == np4 spread over
./docs/performance/scaling/levante_campaign_2026-07-24.md-106-  two nodes (17.12 ms), so node crossing is irrelevant.
./docs/performance/scaling/levante_campaign_2026-07-24.md:107:- HALO VOLUME REFUTED: ghost-cell census on the padded mesh gives
./docs/performance/scaling/levante_campaign_2026-07-24.md-108-  1540/1587/1400/1136 ghost cells per device at np 2/4/8/16 — flat to
./docs/performance/scaling/levante_campaign_2026-07-24.md-109-  falling, and under 3 % of owned cells at every count.
./docs/performance/scaling/levante_campaign_2026-07-24.md:110:- COLLECTIVE COUNT REFUTED (HLO census, ico L7, CPU virtual devices —
./docs/performance/scaling/levante_campaign_2026-07-24.md-111-  device count is a compile-time property so the HLO matches what the GPUs
./docs/performance/scaling/levante_campaign_2026-07-24.md:112:  execute): collective-permutes per step are 3 / 9 / 21 at np 2/4/8, i.e.
./docs/performance/scaling/levante_campaign_2026-07-24.md-113-  np8 issues 2.3x MORE collectives than np4 and still runs 2.5x faster.
./docs/performance/scaling/levante_campaign_2026-07-24.md-114-  Collective COUNT therefore cannot explain the np4 dip (this assumes cost
./docs/performance/scaling/levante_campaign_2026-07-24.md-115-rises with count; a per-message-size effect is not excluded). (Fusion count 136/173/240,
./docs/performance/scaling/levante_campaign_2026-07-24.md-116-  bitcasts 526/582/694 — the np8 program is finer-grained.)
./docs/performance/scaling/levante_campaign_2026-07-24.md:117:- PARTITION METHOD REFUTED (job 26455829): the dip is method-independent —
./docs/performance/scaling/levante_campaign_2026-07-24.md-118-  np4/np8 = 17.22/7.00 ms (sfc), 17.20/6.97 (metis), 19.35/6.26
./docs/performance/scaling/levante_campaign_2026-07-24.md-119-  (geometric). Every method shows the same 2.5-3.1x jump.
./docs/performance/scaling/levante_campaign_2026-07-24.md:120:- XLA CODEGEN ENV KNOBS REFUTED (job 26455948): np4 is 17.12 ms base,
./docs/performance/scaling/levante_campaign_2026-07-24.md-121-  17.06 autotune-level-4, 17.10 latency-hiding-off, 17.01
./docs/performance/scaling/levante_campaign_2026-07-24.md-122-  command-buffers-off — every arm within 1 %, none recovers np4.
./docs/performance/scaling/levante_campaign_2026-07-24.md-123-  (The multi-output-fusion arm errored on an unsupported flag name and is
./docs/performance/scaling/levante_campaign_2026-07-24.md-124-  not counted.)
./docs/performance/scaling/levante_campaign_2026-07-24.md-125-VERDICT: five hypotheses refuted by measurement (placement, halo volume,
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-132-PRACTICAL GUIDANCE MEANWHILE: run this grid at np>=8, where per-device
./docs/performance/scaling/levante_campaign_2026-07-24.md-133-throughput is 304-340 Mc/s/GPU vs 220-248 at np4.
./docs/performance/scaling/levante_campaign_2026-07-24.md-134-
./docs/performance/scaling/levante_campaign_2026-07-24.md-135-RESOLVED 2026-07-26 (nsys job 26479922 + HLO dump 26480096 + sqlite
./docs/performance/scaling/levante_campaign_2026-07-24.md-136-timeline): the dip is an XLA CODEGEN pathology, localized to named
./docs/performance/scaling/levante_campaign_2026-07-24.md:137:kernels. CONFIRMED: (1) the dip reproduces under nsys with matched
./docs/performance/scaling/levante_campaign_2026-07-24.md-138-protocol (L8, padded-16 mesh: 21.01/17.78/7.17 ms at np2/4/8 vs campaign
./docs/performance/scaling/levante_campaign_2026-07-24.md-139-19.90/17.09/6.92 — ~5% profiler overhead); (2) at np4 ONLY, giant
./docs/performance/scaling/levante_campaign_2026-07-24.md-140-serialized "loop fusion" kernels appear — loop_add_fusion_1/2 at 3.6 ms
./docs/performance/scaling/levante_campaign_2026-07-24.md-141-per launch (vs ~3 us for ordinary elementwise kernels) plus a THIRD
./docs/performance/scaling/levante_campaign_2026-07-24.md-142-once-per-step group (the unsuffixed loop_add_fusion: 12 of its 44
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-146-wait): 3.6 + 3.6 + 3.3 ~= 10.5-10.9 ms/step = the np4 excess; (3) in the optimized
./docs/performance/scaling/levante_campaign_2026-07-24.md-147-step HLO these are mega-fusions ON THE HALO PATH: `%loop_add_fusion =
./docs/performance/scaling/levante_campaign_2026-07-24.md-148-f32[491520,26]` (edge-tendency add chain, 22 operands incl. an
./docs/performance/scaling/levante_campaign_2026-07-24.md-149-input_scatter_fusion) and `%loop_add_fusion.4 = f32[163844,26]` (cell
./docs/performance/scaling/levante_campaign_2026-07-24.md-150-array), with the shard_map halo-pack concatenates taking the same adds +
./docs/performance/scaling/levante_campaign_2026-07-24.md:151:parameter lists as operands. PLAUSIBLE (inferred from kInput fusion
./docs/performance/scaling/levante_campaign_2026-07-24.md-152-semantics + operand lists, not separately timed): the emitter RECOMPUTES
./docs/performance/scaling/levante_campaign_2026-07-24.md-153-the expensive scatter+add chain inside each consumer fusion, which is why
./docs/performance/scaling/levante_campaign_2026-07-24.md-154-the cost multiplies. WHY np4: fusion cost-model decisions depend on the
./docs/performance/scaling/levante_campaign_2026-07-24.md-155-shard shape; at np2/np8 the mega-fusion is not built. This also explains
./docs/performance/scaling/levante_campaign_2026-07-24.md-156-why the earlier env-knob sweep missed it — autotune/latency-hiding flags
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-211-multinode improvement measured at LL576 was partly a floor effect, and at
./docs/performance/scaling/levante_campaign_2026-07-24.md-212-a production tile the identical code scales substantially better (0.37 ->
./docs/performance/scaling/levante_campaign_2026-07-24.md-213-0.63). Per-device throughput also rises (259 -> 305 Mc/s/GPU at np4).
./docs/performance/scaling/levante_campaign_2026-07-24.md-214-Config is byte-identical between the two rows; only the grid changes.
./docs/performance/scaling/levante_campaign_2026-07-24.md-215-
./docs/performance/scaling/levante_campaign_2026-07-24.md:216:REFUTED EN ROUTE: the np8 leg timed out twice (>90 min still tracing) while
./docs/performance/scaling/levante_campaign_2026-07-24.md-217-np4 — a LARGER per-device tile — finished in ~25 min, which looked like a
./docs/performance/scaling/levante_campaign_2026-07-24.md-218-compile-time cliff at that device count. It is not: the third attempt ran
./docs/performance/scaling/levante_campaign_2026-07-24.md-219-the identical configuration in **99 seconds** with a 21.6 s compile (job
./docs/performance/scaling/levante_campaign_2026-07-24.md-220-26457693). The earlier hangs were transient/environmental, not
./docs/performance/scaling/levante_campaign_2026-07-24.md-221-reproducible, and no compile-time defect is claimed.
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-246-  differ in machine (A100-80 SXM vs A100-40), jax/tree version and date,
./docs/performance/scaling/levante_campaign_2026-07-24.md-247-  so the ~13% gap is not attributable to any single factor.
./docs/performance/scaling/levante_campaign_2026-07-24.md-248-- Cube "404 vs 141 Mc/s": the 404 is the single-GPU RTX-5090 Held-Suarez
./docs/performance/scaling/levante_campaign_2026-07-24.md-249-  row in `SCALING_SUMMARY.md` SS1; ours is gray+SBM on A100 (job 26445836).
./docs/performance/scaling/levante_campaign_2026-07-24.md-250-  That file's own tier table prices gray+SBM ~3x Held-Suarez, so ~135 is
./docs/performance/scaling/levante_campaign_2026-07-24.md:251:  the expected equivalent vs 141 measured. PLAUSIBLE reconciliation from
./docs/performance/scaling/levante_campaign_2026-07-24.md-252-  two published tables, NOT a matched A/B (GPU, physics and date differ).
./docs/performance/scaling/levante_campaign_2026-07-24.md-253-- Ocean absolutes (A100 f64 165-201 Mc/s, f32 352, job 26445836; 5090 f64
./docs/performance/scaling/levante_campaign_2026-07-24.md-254-  152 / f32 400 from `SCALING_SUMMARY.md` SS1) are of the same order -
./docs/performance/scaling/levante_campaign_2026-07-24.md-255-  again a cross-machine sanity check, not a controlled comparison.
./docs/performance/scaling/levante_campaign_2026-07-24.md-256-- Ginsburg "0.92 eff @2 GPU" reconciled: the old bench silently defaulted
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-350-Bound = per-device compute (32.58 / 14.63 ms) + modelled comm (2.21) +
./docs/performance/scaling/levante_campaign_2026-07-24.md-351-modelled reduction (2.19). The unmodelled gap is **~5 ms/step and roughly
./docs/performance/scaling/levante_campaign_2026-07-24.md-352-FLAT** with device count (7.9 ms at nd2, 6.7 at nd4), which is why the
./docs/performance/scaling/levante_campaign_2026-07-24.md-353-ratio worsens as compute shrinks.
./docs/performance/scaling/levante_campaign_2026-07-24.md-354-
./docs/performance/scaling/levante_campaign_2026-07-24.md:355:WHAT THE GAP IS NOT — the omitted-traffic explanation is REFUTED
./docs/performance/scaling/levante_campaign_2026-07-24.md-356-(`scripts/tmp/probe_ocean_halo_bytes.py`, HLO byte census on CPU virtual
./docs/performance/scaling/levante_campaign_2026-07-24.md-357-devices). The bench's `comm_scope_note` correctly warns that its census is
./docs/performance/scaling/levante_campaign_2026-07-24.md-358-"barotropic implicit-CN PCG scope only … baroclinic 3-D pads NOT counted",
./docs/performance/scaling/levante_campaign_2026-07-24.md-359-and the true volume IS much larger: **16.22 MB/step across 110
./docs/performance/scaling/levante_campaign_2026-07-24.md:360:collective-permutes vs the censused 2.25 MB — a 7.2x undercount**. But
./docs/performance/scaling/levante_campaign_2026-07-24.md-361-completing the census moves the bound by only **0.22 ms**, because the
./docs/performance/scaling/levante_campaign_2026-07-24.md-362-comm term is LATENCY-dominated: at 122 messages x 17.82 us the latency part
./docs/performance/scaling/levante_campaign_2026-07-24.md-363-is 2.174 ms while even 16 MB at 64.22 GB/s is just 0.253 ms.
./docs/performance/scaling/levante_campaign_2026-07-24.md-364-
./docs/performance/scaling/levante_campaign_2026-07-24.md-365-So with the byte census completed the unexplained residual is still 5.5 ms
./docs/performance/scaling/levante_campaign_2026-07-24.md-366-(nd2) and 4.3 ms (nd4).
./docs/performance/scaling/levante_campaign_2026-07-24.md-367-
./docs/performance/scaling/levante_campaign_2026-07-24.md:368:SECOND CANDIDATE ALSO REFUTED (`scripts/tmp/probe_sharded_overhead.py`,
./docs/performance/scaling/levante_campaign_2026-07-24.md-369-job 26458553): the sharded formulation does NOT do measurably more work.
./docs/performance/scaling/levante_campaign_2026-07-24.md-370-Timing the SHARDED step on a 1-device mesh (all the padding, band-edge and
./docs/performance/scaling/levante_campaign_2026-07-24.md-371-v-row-reconstruction machinery present, ppermutes self-to-self so no real
./docs/performance/scaling/levante_campaign_2026-07-24.md-372-traffic) against the UNSHARDED step at the identical tile:
./docs/performance/scaling/levante_campaign_2026-07-24.md-373-
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-383-divided by the message count is **83 us/message at nd2 and 73 us at nd4**,
./docs/performance/scaling/levante_campaign_2026-07-24.md-384-versus **17.8 us** for the same collective measured in isolation — an in-
./docs/performance/scaling/levante_campaign_2026-07-24.md-385-context cost 4-5x the best case. That is consistent with EXPOSED,
./docs/performance/scaling/levante_campaign_2026-07-24.md-386-un-overlapped communication rather than raw wire time.
./docs/performance/scaling/levante_campaign_2026-07-24.md-387-
./docs/performance/scaling/levante_campaign_2026-07-24.md:388:THIRD CANDIDATE REFUTED, AND IT IDENTIFIES THE MECHANISM (job 26458930).
./docs/performance/scaling/levante_campaign_2026-07-24.md-389-If the residual were communication the scheduler is currently hiding work
./docs/performance/scaling/levante_campaign_2026-07-24.md-390-behind, DISABLING XLA's latency-hiding scheduler would hurt. It does not:
./docs/performance/scaling/levante_campaign_2026-07-24.md-391-
./docs/performance/scaling/levante_campaign_2026-07-24.md-392-| arm | nd2 | nd4 | vs default |
./docs/performance/scaling/levante_campaign_2026-07-24.md-393-|---|---|---|---|
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-454-genuinely PER-ITERATION, not a constant misattributed to iterations. But
./docs/performance/scaling/levante_campaign_2026-07-24.md-455-the nd4 intercept EXCEEDS the measured 144-row compute term (14.63 ms) by
./docs/performance/scaling/levante_campaign_2026-07-24.md-456-**1.48 ms**, so a fixed non-PCG overhead does exist and the 60 iterations
./docs/performance/scaling/levante_campaign_2026-07-24.md-457-do NOT explain the entire residual.
./docs/performance/scaling/levante_campaign_2026-07-24.md-458-
./docs/performance/scaling/levante_campaign_2026-07-24.md:459:CODEX OBJECTION (a) TESTED AND REFUTED (job 26459817). Rather than divide
./docs/performance/scaling/levante_campaign_2026-07-24.md-460-the nd1 slope by 4, measure the per-iteration slope directly at each tile
./docs/performance/scaling/levante_campaign_2026-07-24.md-461-on ONE device:
./docs/performance/scaling/levante_campaign_2026-07-24.md-462-
./docs/performance/scaling/levante_campaign_2026-07-24.md-463-| tile | measured us/iter |
./docs/performance/scaling/levante_campaign_2026-07-24.md-464-|---|---|
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-640-
./docs/performance/scaling/levante_campaign_2026-07-24.md-641-Per-step time is flat 100 -> 600 steps (12.81 -> 12.74 ms). Both arms' heat
./docs/performance/scaling/levante_campaign_2026-07-24.md-642-drift grows ~linearly and is similar between them, consistent with the
./docs/performance/scaling/levante_campaign_2026-07-24.md-643-shared baroclinic/tracer path dominating it.
./docs/performance/scaling/levante_campaign_2026-07-24.md-644-
./docs/performance/scaling/levante_campaign_2026-07-24.md:645:MECHANISM CONFIRMED FROM A THIRD ANGLE. The wide-halo arm records its own
./docs/performance/scaling/levante_campaign_2026-07-24.md-646-message census: **120 standard barotropic messages/step -> 4** (n_loop=30
./docs/performance/scaling/levante_campaign_2026-07-24.md-647-substeps, stencil reach 3, one fixed wide exchange per chunk). A 30x cut in
./docs/performance/scaling/levante_campaign_2026-07-24.md-648-barotropic exchanges is precisely why it wins where sync dominates, and it
./docs/performance/scaling/levante_campaign_2026-07-24.md-649-is the SAME quantity the single_reduce analysis isolated as the half it
./docs/performance/scaling/levante_campaign_2026-07-24.md-650-could not touch (44.1 us/iter of matvec halo). Three independent
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-982-
./docs/performance/scaling/levante_campaign_2026-07-24.md-983-## Cube optimisation: bounded BEFORE implementing — and the bound killed the plan
./docs/performance/scaling/levante_campaign_2026-07-24.md-984-
./docs/performance/scaling/levante_campaign_2026-07-24.md-985-Directive was to push the cube toward its limit. Codex round-16 defined
./docs/performance/scaling/levante_campaign_2026-07-24.md-986-the strategy first (per the standing pre-implementation rule) and its
./docs/performance/scaling/levante_campaign_2026-07-24.md:987:cheapest-bound step then REFUTED the intervention I was about to build.
./docs/performance/scaling/levante_campaign_2026-07-24.md-988-
./docs/performance/scaling/levante_campaign_2026-07-24.md-989-**Codex corrections to my reading:**
./docs/performance/scaling/levante_campaign_2026-07-24.md-990-* My "88 SendRecv / 4 rounds = 22 exchanges" was wrong. The kt=2 tile pad
./docs/performance/scaling/levante_campaign_2026-07-24.md-991-  emits **16 phases per logical scalar pad** (4 edges + 4 guards + 4
./docs/performance/scaling/levante_campaign_2026-07-24.md-992-  diagonals + 4 corner slivers) for serial-exact offset/corner handling
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-1531-   CLOSED as not-worth-building, with receipts. (Also moot for the SPMD
./docs/performance/scaling/levante_campaign_2026-07-24.md-1532-   lane: jax equal-shard sharding would need padding to the max band,
./docs/performance/scaling/levante_campaign_2026-07-24.md-1533-   returning exactly the imbalance removed.)
./docs/performance/scaling/levante_campaign_2026-07-24.md-1534-
./docs/performance/scaling/levante_campaign_2026-07-24.md-1535-   *Expensive half — gather/scatter compaction: MEASURED, and the audit's
./docs/performance/scaling/levante_campaign_2026-07-24.md:1536:   "~2x" is REFUTED* (`bench_gather_vs_slice_stencil.py`, job 26479884,
./docs/performance/scaling/levante_campaign_2026-07-24.md-1537-   A100 f32, correctness self-checked). Per-cell gather penalty for a
./docs/performance/scaling/levante_campaign_2026-07-24.md-1538-   5-point Laplacian vs the dense sliced version: **1.40-1.77x**, so
./docs/performance/scaling/levante_campaign_2026-07-24.md-1539-   compaction wins only when wet_fraction < 0.56-0.72 (size-dependent).
./docs/performance/scaling/levante_campaign_2026-07-24.md-1540-   At the REAL global-ocean wet fraction (~0.71), packed-gather is a net
./docs/performance/scaling/levante_campaign_2026-07-24.md-1541-   LOSS on the full LL576 grid (ratio 1.16) and a wash at the nd4 tile
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-1628-  METIS was NOT tested).
./docs/performance/scaling/levante_campaign_2026-07-24.md-1629-* **`block:cyclic` stays mandatory on packed CPU lanes** (E/D = 1.61x at
./docs/performance/scaling/levante_campaign_2026-07-24.md-1630-  a byte-identical partition). NOTE the second `--distribution` field is
./docs/performance/scaling/levante_campaign_2026-07-24.md-1631-  the INTRA-NODE (socket) distribution — both arms place ranks on nodes
./docs/performance/scaling/levante_campaign_2026-07-24.md-1632-  identically; the swing is socket-level. Mechanism (per-socket
./docs/performance/scaling/levante_campaign_2026-07-24.md:1633:  memory-bandwidth balance) PLAUSIBLE, consistent with the np16 Milan
./docs/performance/scaling/levante_campaign_2026-07-24.md-1634-  2.13x receipt; never instrumented with bandwidth counters.
./docs/performance/scaling/levante_campaign_2026-07-24.md-1635-* Caveats: timing-only receipt — no parity/conservation gate ran in
./docs/performance/scaling/levante_campaign_2026-07-24.md-1636-  these arms, and the CPU nodes emit `UCX WARN transports
./docs/performance/scaling/levante_campaign_2026-07-24.md-1637-  'cuda_copy','cuda_ipc','gdr_copy' are not available` (the _env.sh GPU
./docs/performance/scaling/levante_campaign_2026-07-24.md-1638-  UCX_TLS list on a CPU node; UCX falls back to rc/sm — cosmetic for
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-1643-f32, sfc partition (padded-128 reorder), lloyd=0 LABELLED SYNTHETIC
./docs/performance/scaling/levante_campaign_2026-07-24.md-1644-scaling mesh, executed padded n_cells = 2,621,568 (natural 2,621,442),
./docs/performance/scaling/levante_campaign_2026-07-24.md-1645-L26; steps 12 / warmup 3; physics=none dynamics-only bench. Provenance:
./docs/performance/scaling/levante_campaign_2026-07-24.md-1646-np64 and np128 rows record `git_sha: 7151d12a1`; the np32 row's field
./docs/performance/scaling/levante_campaign_2026-07-24.md-1647-reads `unknown` — same allocation, same submitted script, so the same
./docs/performance/scaling/levante_campaign_2026-07-24.md:1648:binary is PLAUSIBLE but that row stays non-reproduction-grade on its
./docs/performance/scaling/levante_campaign_2026-07-24.md-1649-own (codex r20/r21):
./docs/performance/scaling/levante_campaign_2026-07-24.md-1650-
./docs/performance/scaling/levante_campaign_2026-07-24.md-1651-| GPUs | cells/GPU | ms/step | GC/s (cell-levels) |
./docs/performance/scaling/levante_campaign_2026-07-24.md-1652-|---|---|---|---|
./docs/performance/scaling/levante_campaign_2026-07-24.md-1653-| 32 | 81.9k | 12.47 | 5.47 |
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-1678-   sets vs SAME-JOB solo controls bracketing phase B (solo before AND
./docs/performance/scaling/levante_campaign_2026-07-24.md-1679-   after — BRACKETED, not fully counterbalanced; a penalty's attribution
./docs/performance/scaling/levante_campaign_2026-07-24.md-1680-   to fabric vs placement/drift needs the per-step nodelist table +
./docs/performance/scaling/levante_campaign_2026-07-24.md-1681-   follow-up). steps=5000 so the stepping window
./docs/performance/scaling/levante_campaign_2026-07-24.md-1682-   (~60 s) dwarfs launch skew; per-arm `SLURM_STEP_NODELIST` +
./docs/performance/scaling/levante_campaign_2026-07-24.md:1683:   wall-clock brackets logged as overlap evidence. CONFIRM bar:
./docs/performance/scaling/levante_campaign_2026-07-24.md-1684-   max(replica) <= 1.10x mean(solo) => guaranteed aggregate >= 3.64x the
./docs/performance/scaling/levante_campaign_2026-07-24.md-1685-   32-GPU solo rate (>= 19.9 GC/s if solo reproduces 5.47) = ~3.3x the
./docs/performance/scaling/levante_campaign_2026-07-24.md:1686:   observed 128-GPU single-trajectory rate. REFUTE: replica slowdown
./docs/performance/scaling/levante_campaign_2026-07-24.md-1687-   >10 % = a CO-EXECUTION penalty, quantified per replica — its
./docs/performance/scaling/levante_campaign_2026-07-24.md-1688-   attribution (fabric contention vs placement/topology vs drift) is a
./docs/performance/scaling/levante_campaign_2026-07-24.md-1689-   follow-up, not a conclusion of this job.
./docs/performance/scaling/levante_campaign_2026-07-24.md-1690-2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
./docs/performance/scaling/levante_campaign_2026-07-24.md-1691-   (81.9k/41.0k/20.5k cells/GPU) on the SAME lloyd=0 family, same sfc +
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-1743-
./docs/performance/scaling/levante_campaign_2026-07-24.md-1744-Distribution verified against the masquerade trap: result rows carry
./docs/performance/scaling/levante_campaign_2026-07-24.md-1745-`n_ranks: 512` (the JSON's `metadata.process_count: 1` is the jax-LOCAL
./docs/performance/scaling/levante_campaign_2026-07-24.md-1746-count on this mpi4jax lane, not the world size). 128->256 is
./docs/performance/scaling/levante_campaign_2026-07-24.md-1747-SUPERLINEAR (2.23x for 2x) — classic per-rank working-set cache
./docs/performance/scaling/levante_campaign_2026-07-24.md:1748:transition on Milan (mechanism PLAUSIBLE, uninstrumented). End-to-end
./docs/performance/scaling/levante_campaign_2026-07-24.md-1749-64->512 eff 0.83 at 1k cols/rank: TIMING-ONLY evidence that the lat-lon
./docs/performance/scaling/levante_campaign_2026-07-24.md-1750-CPU lane scales into the hundreds. QUALIFIER (codex r22): the run logs
./docs/performance/scaling/levante_campaign_2026-07-24.md-1751-an out-of-tested-range mpi4jax==0.9.0 pairing ("may fail or produce
./docs/performance/scaling/levante_campaign_2026-07-24.md-1752-incorrect results", parallel/reductions.py runtime check) and UCX
./docs/performance/scaling/levante_campaign_2026-07-24.md-1753-VM_UNMAP warnings — no parity/conservation gate ran, so this ladder is
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-1772-|---|---|---|---|---|
./docs/performance/scaling/levante_campaign_2026-07-24.md-1773-| 81.9k | np8 6.58 | np32 12.47 | 1.895 | **0.53** |
./docs/performance/scaling/levante_campaign_2026-07-24.md-1774-| 41.0k | np16 6.43 | np64 9.60 | 1.493 | 0.67 |
./docs/performance/scaling/levante_campaign_2026-07-24.md-1775-| 20.5k | np32 7.29 | np128 11.48 | 1.575 | 0.64 |
./docs/performance/scaling/levante_campaign_2026-07-24.md-1776-
./docs/performance/scaling/levante_campaign_2026-07-24.md:1777:* The falsifiability block's CONFIRM branch fires: ratios stay well
./docs/performance/scaling/levante_campaign_2026-07-24.md-1778-  above 1 with the known Lloyd-family mismatch REMOVED. (This does not
./docs/performance/scaling/levante_campaign_2026-07-24.md-1779-  prove the old confound "only" biased the size — these are
./docs/performance/scaling/levante_campaign_2026-07-24.md-1780-  unreplicated single runs from separate allocations, one comparator
./docs/performance/scaling/levante_campaign_2026-07-24.md-1781-  without row-level provenance; the confounded draft read
./docs/performance/scaling/levante_campaign_2026-07-24.md-1782-  1.80/1.35/1.41 vs 1.90/1.49/1.57 here, and the production-mesh s8
./docs/performance/scaling/levante_campaign_2026-07-24.md-1783-  np8 was 6.92 vs lloyd-0 6.58, -4.9 %, so the mesh family does shift
./docs/performance/scaling/levante_campaign_2026-07-24.md-1784-  absolutes.)
./docs/performance/scaling/levante_campaign_2026-07-24.md-1785-* Restated: at NEAR-matched per-GPU tile, ~quadrupling devices+problem
./docs/performance/scaling/levante_campaign_2026-07-24.md-1786-  costs 1.5-1.9x on this lane — the GPU-side analogue of the ocean CPU
./docs/performance/scaling/levante_campaign_2026-07-24.md-1787-  scale-out term. Weak efficiency 0.53-0.67 at 4x. Mechanism still
./docs/performance/scaling/levante_campaign_2026-07-24.md:1788:  UNATTRIBUTED (PLAUSIBLE candidates unchanged: inter-node neighbour
./docs/performance/scaling/levante_campaign_2026-07-24.md-1789-  fraction growth, collective latency vs count, sfc partition-quality
./docs/performance/scaling/levante_campaign_2026-07-24.md-1790-  decay with parts; the metis receipt argues against pure
./docs/performance/scaling/levante_campaign_2026-07-24.md-1791-  partition-cut explanations, on the CPU lane at least).
./docs/performance/scaling/levante_campaign_2026-07-24.md-1792-* The non-monotone tile dependence of the ratio (largest at the
./docs/performance/scaling/levante_campaign_2026-07-24.md-1793-  LARGEST tile, 1.90 at 81.9k) is unexplained; recorded, not theorised.
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-1797-Closed the bench's own honest-null bound gap (audit item 4) for the
./docs/performance/scaling/levante_campaign_2026-07-24.md-1798-lat-lon lane, using only repo instruments:
./docs/performance/scaling/levante_campaign_2026-07-24.md-1799-
./docs/performance/scaling/levante_campaign_2026-07-24.md-1800-* **Halo census** (new probe `scripts/tmp/_probe_latlon_halo_census.py`,
./docs/performance/scaling/levante_campaign_2026-07-24.md-1801-  virtual-CPU forced-host-platform lowering of the REAL
./docs/performance/scaling/levante_campaign_2026-07-24.md:1802:  `make_sharded_atm_latlon_step`): **41 collective-permutes + 1
./docs/performance/scaling/levante_campaign_2026-07-24.md-1803-  all-reduce per step**, nd-INDEPENDENT (identical at nd=8 and nd=16 —
./docs/performance/scaling/levante_campaign_2026-07-24.md-1804-  the 1-D band structure check). Exact CP payload from compiled-HLO
./docs/performance/scaling/levante_campaign_2026-07-24.md-1805-  result shapes: 4,635,408 B/dev/step at n_lon=1024 L26 f32 = 1.06x the
./docs/performance/scaling/levante_campaign_2026-07-24.md-1806-  single-row slab model; linear in n_lon (checked 1024 vs 2048, 0.07 %
./docs/performance/scaling/levante_campaign_2026-07-24.md-1807-  residual) -> **18.5 MB/dev/step at n_lon=4096 f32**. CAVEAT: CPU
./docs/performance/scaling/levante_campaign_2026-07-24.md-1808-  lowering; GPU-side collective combining could change the executed
./docs/performance/scaling/levante_campaign_2026-07-24.md-1809-  count (metadata.py:214) — the bound is a MODEL.
./docs/performance/scaling/levante_campaign_2026-07-24.md:1810:* **Same-tile nd=1 compute baselines** (job 26630370, roofline recipe):
./docs/performance/scaling/levante_campaign_2026-07-24.md-1811-  16x4096 f32 1.659 ms, 32x4096 f32 2.837, 16x4096 f64 2.973.
./docs/performance/scaling/levante_campaign_2026-07-24.md-1812-  Approximation, recorded: nd=1 includes pole tiles -> compute term
./docs/performance/scaling/levante_campaign_2026-07-24.md-1813-  biased HIGH -> bound conservative.
./docs/performance/scaling/levante_campaign_2026-07-24.md:1814:* **Calibrated bound** (`metadata.calibrated_bound`, measured fabric
./docs/performance/scaling/levante_campaign_2026-07-24.md-1815-  constants: IB 26.3 us / 23.5 GB/s, NVLink 17.8 / 64.2):
./docs/performance/scaling/levante_campaign_2026-07-24.md-1816-
./docs/performance/scaling/levante_campaign_2026-07-24.md-1817-| row | measured | t_bound (IB) | measured/bound |
./docs/performance/scaling/levante_campaign_2026-07-24.md-1818-|---|---|---|---|
./docs/performance/scaling/levante_campaign_2026-07-24.md-1819-| LL2048@64 f32 | 6.732 | 2.863 | **2.35** |
./docs/performance/scaling/levante_campaign_2026-07-24.md-1820-| LL2048@128 f32 | 5.577 | 1.848 | **3.02** |
./docs/performance/scaling/levante_campaign_2026-07-24.md-1821-| LL2048@128 f64 | 9.602 | 2.999 | **3.20** |
./docs/performance/scaling/levante_campaign_2026-07-24.md-1822-
./docs/performance/scaling/levante_campaign_2026-07-24.md:1823:* **The lat-lon GPU panel sits ~2.4-3.2x ABOVE its modeled limit** — the
./docs/performance/scaling/levante_campaign_2026-07-24.md-1824-  eff-0.60 strong leg is NOT close to the fabric+compute floor. Leading
./docs/performance/scaling/levante_campaign_2026-07-24.md:1825:  PLAUSIBLE mechanism (uninstrumented): effective per-CP cost
./docs/performance/scaling/levante_campaign_2026-07-24.md-1826-  (launch + schedule + stream sync) well above the raw 26 us fabric
./docs/performance/scaling/levante_campaign_2026-07-24.md-1827-  latency across 41 dependency-chained exchanges — the arXiv:2607.16100
./docs/performance/scaling/levante_campaign_2026-07-24.md-1828-  small-collective regime. The model itself notes the serialized-latency
./docs/performance/scaling/levante_campaign_2026-07-24.md-1829-  vs overlap biases pull opposite ways; treat measured/bound as a
./docs/performance/scaling/levante_campaign_2026-07-24.md-1830-  consistency diagnostic, not proven headroom.
./docs/performance/scaling/levante_campaign_2026-07-24.md:1831:* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
./docs/performance/scaling/levante_campaign_2026-07-24.md-1832-  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
./docs/performance/scaling/levante_campaign_2026-07-24.md-1833-  same-job control, falsifiability block in the script. The ocean-lane
./docs/performance/scaling/levante_campaign_2026-07-24.md-1834-  null for these flags was reduction-dominated — first atm test.
--
./docs/performance/scaling/scaling_gpu.md-795-geostrophic-adjustment eta amplification is similar across solvers.
./docs/performance/scaling/scaling_gpu.md-796-
./docs/performance/scaling/scaling_gpu.md-797-PR #319 description updated to reflect iter 21-25 corrections
./docs/performance/scaling/scaling_gpu.md-798-(unit-bug walkback, CFL-validated nsub recommendations, final ladder).
./docs/performance/scaling/scaling_gpu.md-799-
./docs/performance/scaling/scaling_gpu.md:800:### Iter 25 — 2026-05-27 — MPAS CFL stability sweep CONFIRMS iter-22 nsub=10
./docs/performance/scaling/scaling_gpu.md-801-
./docs/performance/scaling/scaling_gpu.md-802-Mirror iter-24 check on MPAS. I5 fp64 dt=600 s, 200 steps (33 h) with
./docs/performance/scaling/scaling_gpu.md-803-eta=0.1 m kick:
./docs/performance/scaling/scaling_gpu.md-804-
./docs/performance/scaling/scaling_gpu.md-805-| nsub | finite? | |eta|_max  |
--
./docs/performance/scaling/cube_production_tiling_design.md-135-  `test_tiled_fv3_sw_full.py` (job 8482569: TILED_FULL_GATE_OK). The production
./docs/performance/scaling/cube_production_tiling_design.md-136-  cube SW dycore tendency now runs on np=6*kt^2 devices.
./docs/performance/scaling/cube_production_tiling_design.md-137-- **MULTI-NODE VALIDATION** — `scripts/validate/validate_tiled_fv3_sw_multinode.py`
./docs/performance/scaling/cube_production_tiling_design.md-138-  + `scripts/cluster/scaling_ginsburg/tiled_fv3_sw_multinode.sbatch`: runs the
./docs/performance/scaling/cube_production_tiling_design.md-139-  full stage under REAL multi-controller `jax.distributed` across 2 nodes (np24,
./docs/performance/scaling/cube_production_tiling_design.md:140:  in-stage ppermutes -> cross-NODE collective-permute); each process
./docs/performance/scaling/cube_production_tiling_design.md-141-  self-validates its local tile vs serial (rel<1e-9). Correctness, not a bench.
./docs/performance/scaling/cube_production_tiling_design.md-142-- **PRODUCTION ASSEMBLY WIRED (2026-07-09)** — the blocked persistent step
./docs/performance/scaling/cube_production_tiling_design.md-143-  (`make_tiled_fv3_hydrostatic_step_blocked_2d`, input layout == output
./docs/performance/scaling/cube_production_tiling_design.md-144-  layout, in-stage telescoping mass fixer, optional moist column physics)
./docs/performance/scaling/cube_production_tiling_design.md-145-  + `make_tiled_cc_loop` (adapter enter/step/exit_) + the
--
./docs/performance/scaling/literature_parallelization_2026-06.md-58-   is a census-driven trimming lever for the tiled stage.
./docs/performance/scaling/literature_parallelization_2026-06.md-59-7. **GSPMD-auto can silently insert a pathological AllGather consuming 80%
./docs/performance/scaling/literature_parallelization_2026-06.md-60-   of runtime; the prescribed detection is the device-profile timeline**
./docs/performance/scaling/literature_parallelization_2026-06.md-61-   (jax-ml scaling book — 3-0). Independently confirms our np24
./docs/performance/scaling/literature_parallelization_2026-06.md-62-   GSPMD-auto 39× story + HLO-census discipline. (The JEP claim that
./docs/performance/scaling/literature_parallelization_2026-06.md:63:   shard_map is "only a surgical escape hatch" was REFUTED 1-2; the
./docs/performance/scaling/literature_parallelization_2026-06.md-64-   scaling book positions explicit collectives as first-class for
./docs/performance/scaling/literature_parallelization_2026-06.md-65-   comm-critical code.)
./docs/performance/scaling/literature_parallelization_2026-06.md-66-8. **Manual comm/compute overlap in shard_map works:** stepwise
./docs/performance/scaling/literature_parallelization_2026-06.md-67-   ppermute+partial-compute "collective matmul" removed ~77% of the
./docs/performance/scaling/literature_parallelization_2026-06.md-68-   communication overhead vs a blocking AllGather (244 µs vs 311 µs, 224 µs
./docs/performance/scaling/literature_parallelization_2026-06.md-69-   unsharded baseline) (3-0). The lighter sibling of our parked deep-halo
./docs/performance/scaling/literature_parallelization_2026-06.md-70-   idea — applicable to halo+interior-stencil overlap in the tiled stage.
./docs/performance/scaling/literature_parallelization_2026-06.md:71:   (The JEP's own transformer overlap example claim was REFUTED 0-3 — cite
./docs/performance/scaling/literature_parallelization_2026-06.md-72-   the scaling book, not the JEP, for this pattern.)
./docs/performance/scaling/literature_parallelization_2026-06.md-73-9. **CPU same-node wall is universal, not a legoESM defect.** Veros/JAX on
./docs/performance/scaling/literature_parallelization_2026-06.md-74-   8×32-core CPU nodes: within 1.4× of Fortran+MPI, the gap attributed to
./docs/performance/scaling/literature_parallelization_2026-06.md-75-   DRAM-bandwidth-bound execution + XLA's incomplete fusion; at high counts
./docs/performance/scaling/literature_parallelization_2026-06.md-76-   communication + barotropic dominate (3-0). Keeps our 1-proc/node +
--
scripts/run/run_rce_mpi_long.py-517-    )
scripts/run/run_rce_mpi_long.py-518-
scripts/run/run_rce_mpi_long.py-519-    # iter-181 theta' noise seed in the lowest 4 levels (RCEMIP /
scripts/run/run_rce_mpi_long.py-520-    # Wing 2018 standard symmetry-breaker). iter-212 update: the
scripts/run/run_rce_mpi_long.py-521-    # original iter-181 claim that theta' was safer than qv noise
scripts/run/run_rce_mpi_long.py:522:    # because it doesn't enter the LW optical depth was REFUTED at
scripts/run/run_rce_mpi_long.py-523-    # the iter-212 smoke — theta' nonzero amplitudes blow up via a
scripts/run/run_rce_mpi_long.py-524-    # DIFFERENT mechanism (direct buoyancy injection, not radiation
scripts/run/run_rce_mpi_long.py-525-    # feedback). Same F11 dx=4 km wall, just from a different
scripts/run/run_rce_mpi_long.py-526-    # physics pathway.
scripts/run/run_rce_mpi_long.py-527-    # iter-203: added --theta-noise-mode to pick the perturbation
--
./docs/science/specs/new_test_dycores.md-122-
./docs/science/specs/new_test_dycores.md-123-## State after iter-1..140 (compressed at iter-140)
./docs/science/specs/new_test_dycores.md-124-
./docs/science/specs/new_test_dycores.md-125-**iter-131..140 highlights — TC2/TC3 cube blowup investigation phase**:
./docs/science/specs/new_test_dycores.md-126-
./docs/science/specs/new_test_dycores.md:127:- **iter-132 TC3 cube comment upgraded** from pre-emptive CAUTION to **CONFIRMED** full-mode BLOWUP warning (iter-123 verified TC3 cube FAILs at step 2250).
./docs/science/specs/new_test_dycores.md-128-- **iter-135 TC2 quick 3-grid parity verified**: cube 0.3177 / ico 0.3552 / spec 0.3597 (all within 12 %; cube is LOWEST at quick mode).
./docs/science/specs/new_test_dycores.md-129-- **iter-136 FV3 oracle audit (n_split)**: FV3 control for mountain test is `n_split=10`.  Our `n_acoustic_substeps=20` is already 2× FV3 — so **hypothesis 1** (increase acoustic substeps) is unlikely to be the fix.  iter-104 hypothesis list updated.
./docs/science/specs/new_test_dycores.md-130-- **iter-138 hypothesis 2 PROBE in progress**: TC2 cube with 16× hyperdiff (vs baseline 4×), running --days 0.15 (3.6 hr past current blowup at day 0.13).  Expected wall ~67 min.  Currently 11 min in.  Result iter-141+.
./docs/science/specs/new_test_dycores.md-131-- **iter-133 TC3 quick refresh FAILED twice** (~25 min stuck each).  TC3 quick parity vs iter-7 doc claim remains UNVERIFIED but full-mode FAIL is the critical finding (already documented).
./docs/science/specs/new_test_dycores.md-132-- **iter-130 doc compression** of iter-121..130 — NH matrix completion phase.
--
./docs/science/specs/CRM_implementation.md-151-| dt=20 weno5 β=0.2 | 0 | 0.1 K | step 50 | ~40%/step |
./docs/science/specs/CRM_implementation.md-152-| (any) | 0 | 0 | stable indefinitely | — |
./docs/science/specs/CRM_implementation.md-153-| (any, --no-radiation) | nonzero qv | 0 | stable | — |
./docs/science/specs/CRM_implementation.md-154-| dt=20 vl β=0.2 --no-radiation smooth_k1 | 0 | 0.01 K | **step 20** | ~80%/step |
./docs/science/specs/CRM_implementation.md-155-
./docs/science/specs/CRM_implementation.md:156:iter-212 update: the bottom row REFUTES the prior iter-181 claim that "theta' is safer than qv because it doesn't enter the LW optical depth". theta' noise also destabilises *without* radiation, via direct buoyancy injection rather than radiation feedback. Same F11 dx=4 km wall, different physics pathway. Only qv noise specifically requires radiation to destabilise.
./docs/science/specs/CRM_implementation.md-157-
./docs/science/specs/CRM_implementation.md-158-Diagnostic — `/tmp/diag_rad_qv.py` calls `gray_radiation` directly on a column-symmetric IC and on an IC with one column perturbed by +1e-8 kg/kg in the lowest 4 levels:
./docs/science/specs/CRM_implementation.md-159-* Column-symmetric: heating rate range −3.6e-5 → +2.2e-5 K/s. Standard gray-RCE pattern (LW cooling above z~17 km absorption peak, warming below).
./docs/science/specs/CRM_implementation.md-160-* Perturbed: heating-rate spread between perturbed and unperturbed columns = **3.99e-12 K/s** at the perturbed location (z=3850 m, in the absorption band). Linear-and-tiny — the radiation IS responding correctly to the noise.
./docs/science/specs/CRM_implementation.md-161-
--
./scripts/bench/bench_mpas_spmd_scaling.py-413-
./scripts/bench/bench_mpas_spmd_scaling.py-414-    if jax.process_count() > 1:
./scripts/bench/bench_mpas_spmd_scaling.py-415-        from jax.experimental import multihost_utils
./scripts/bench/bench_mpas_spmd_scaling.py-416-        multihost_utils.sync_global_devices("mpas_spmd_bench_end")
./scripts/bench/bench_mpas_spmd_scaling.py-417-
./scripts/bench/bench_mpas_spmd_scaling.py:418:    # HLO collective-permute census (#1113 ask 2): a STATIC compile property of
./scripts/bench/bench_mpas_spmd_scaling.py-419-    # the sharded step — the ppermute ROUND count that decomposes multi-node
./scripts/bench/bench_mpas_spmd_scaling.py-420-    # overhead (overhead ~= CPs/step * ~0.11 ms launch floor). The cube benches
./scripts/bench/bench_mpas_spmd_scaling.py-421-    # record this; the MPAS row did not, forcing an out-of-band census. Counted
./scripts/bench/bench_mpas_spmd_scaling.py-422-    # AFTER the timed loop so the census compile can't perturb per_step_ms[0]'s
./scripts/bench/bench_mpas_spmd_scaling.py-423-    # compile timing (the executable is already cached — this re-lower/compile
--
./scripts/bench/bench_atm_latlon_spmd_scaling.py-76-# under rec["metadata"] so a virtual-CPU-device proxy, a gloo/TCP fabric run,
./scripts/bench/bench_atm_latlon_spmd_scaling.py-77-# or an f32 ablation is falsifiable from the JSONL row alone.  metadata.py
./scripts/bench/bench_atm_latlon_spmd_scaling.py-78-# imports JAX lazily, so this is safe before jax.distributed.initialize.
./scripts/bench/bench_atm_latlon_spmd_scaling.py-79-from metadata import (  # noqa: E402
./scripts/bench/bench_atm_latlon_spmd_scaling.py-80-    annotate_incomplete,
./scripts/bench/bench_atm_latlon_spmd_scaling.py:81:    calibrated_bound,
./scripts/bench/bench_atm_latlon_spmd_scaling.py-82-    comm_accounting,
./scripts/bench/bench_atm_latlon_spmd_scaling.py-83-    scaling_metadata,
./scripts/bench/bench_atm_latlon_spmd_scaling.py-84-    tidy_throughput_fields,
./scripts/bench/bench_atm_latlon_spmd_scaling.py-85-)
./scripts/bench/bench_atm_latlon_spmd_scaling.py-86-
--
./scripts/bench/bench_atm_latlon_spmd_scaling.py-341-        bytes_per_message=_bytes_msg,
./scripts/bench/bench_atm_latlon_spmd_scaling.py-342-        full_state_gathers_per_step=0,   # fused scan/segment: no per-step gather
./scripts/bench/bench_atm_latlon_spmd_scaling.py-343-        scope_note=_comm_note,
./scripts/bench/bench_atm_latlon_spmd_scaling.py-344-        bytes_are_lower_bound=_bytes_lower,
./scripts/bench/bench_atm_latlon_spmd_scaling.py-345-    )
./scripts/bench/bench_atm_latlon_spmd_scaling.py:346:    bound_rec = calibrated_bound(
./scripts/bench/bench_atm_latlon_spmd_scaling.py-347-        measured_fused_step_ms=_measured_med,
./scripts/bench/bench_atm_latlon_spmd_scaling.py-348-        # Invalid rows feed the bound NOTHING: even the CLI-provided nd=1
./scripts/bench/bench_atm_latlon_spmd_scaling.py-349-        # baseline is withheld so bound_ingredients.compute_ms cannot dress
./scripts/bench/bench_atm_latlon_spmd_scaling.py-350-        # a diverging row up as a modelled one (codex).
./scripts/bench/bench_atm_latlon_spmd_scaling.py-351-        single_device_fused_step_ms=(
--
./scripts/bench/run_scaling_diagnosis.py-746-            if c is None:
./scripts/bench/run_scaling_diagnosis.py-747-                print("  census: unavailable (step not lowerable on this "
./scripts/bench/run_scaling_diagnosis.py-748-                      "backend)")
./scripts/bench/run_scaling_diagnosis.py-749-            else:
./scripts/bench/run_scaling_diagnosis.py-750-                print(f"  {census_info['n_devices']} device(s): "
./scripts/bench/run_scaling_diagnosis.py:751:                      f"{c['collective_permute']} collective-permute, "
./scripts/bench/run_scaling_diagnosis.py-752-                      f"{c['all_reduce']} all-reduce, "
./scripts/bench/run_scaling_diagnosis.py-753-                      f"{c['all_gather']} all-gather / step")
./scripts/bench/run_scaling_diagnosis.py-754-                if census_info["n_devices"] == 1:
./scripts/bench/run_scaling_diagnosis.py-755-                    print("  (single device: no inter-device schedule — run "
./scripts/bench/run_scaling_diagnosis.py-756-                          "with XLA_FLAGS=--xla_force_host_platform_device_"
--
./scripts/bench/roofline_probe.py-920-    ``run_levante_gpu_scaling.TimingResult`` (``hlo_collective_permute*``)
./scripts/bench/roofline_probe.py-921-    — this reporter does NOT re-derive the census, it consumes the one the
./scripts/bench/roofline_probe.py-922-    timed-executable HLO guard already produced.
./scripts/bench/roofline_probe.py-923-
./scripts/bench/roofline_probe.py-924-    Collective time-floor: ``collective_latency_ms`` × (number of
./scripts/bench/roofline_probe.py:925:    collective ops in the step).  We count each sync collective-permute
./scripts/bench/roofline_probe.py-926-    plus each async start as one round (done ops are the completion half
./scripts/bench/roofline_probe.py-927-    of a start and are not double-counted).
./scripts/bench/roofline_probe.py-928-
./scripts/bench/roofline_probe.py-929-    PCG Amdahl: if ``pcg_iterations`` (M) is given, evaluate
./scripts/bench/roofline_probe.py-930-    ``T = local_stencil + M*(2*allreduce_lat + halo + stencil)`` and report
--
./scripts/bench/bench_cube_tiled_step_scaling.py-12-envelope, ``tiled_step_adapter._refuse``).  The full production driver keeps
./scripts/bench/bench_cube_tiled_step_scaling.py-13-its loud "tiled dycore unwired (P4)" warning; this lane is where >6-GPU
./scripts/bench/bench_cube_tiled_step_scaling.py-14-production stepping is measured TODAY.
./scripts/bench/bench_cube_tiled_step_scaling.py-15-
./scripts/bench/bench_cube_tiled_step_scaling.py-16-Anti-fake-scaling guards:
./scripts/bench/bench_cube_tiled_step_scaling.py:17:  * compiled-HLO census: the step must contain collective-permutes and NO
./scripts/bench/bench_cube_tiled_step_scaling.py-18-    full-cube all-gather (``find_fullcube_allgathers`` — an all-gather means
./scripts/bench/bench_cube_tiled_step_scaling.py-19-    replicated, not tiled, execution): the row is REFUSED otherwise;
./scripts/bench/bench_cube_tiled_step_scaling.py-20-  * shared metadata v2 rows (virtual-CPU devices flagged; transport
./scripts/bench/bench_cube_tiled_step_scaling.py-21-    auto-resolves) — a CPU smoke row can never masquerade as GPU scaling;
./scripts/bench/bench_cube_tiled_step_scaling.py-22-  * --parity-gate: ONE tiled step vs one serial untiled step at the adapter
--
./scripts/bench/bench_cube_tiled_step_scaling.py-226-    # SAME audited executable — the message-count = latency-bound lever.
./scripts/bench/bench_cube_tiled_step_scaling.py-227-    hlo_census = count_collectives(hlo)
./scripts/bench/bench_cube_tiled_step_scaling.py-228-    allgathers = find_fullcube_allgathers(hlo, n=args.resolution)
./scripts/bench/bench_cube_tiled_step_scaling.py-229-    if n_ppermute == 0:
./scripts/bench/bench_cube_tiled_step_scaling.py-230-        raise SystemExit(
./scripts/bench/bench_cube_tiled_step_scaling.py:231:            "compiled tiled step contains NO collective-permutes — the "
./scripts/bench/bench_cube_tiled_step_scaling.py-232-            "halos did not tile (replicated execution); refusing to "
./scripts/bench/bench_cube_tiled_step_scaling.py-233-            "record a fake scaling row.")
./scripts/bench/bench_cube_tiled_step_scaling.py-234-    if allgathers:
./scripts/bench/bench_cube_tiled_step_scaling.py-235-        raise SystemExit(
./scripts/bench/bench_cube_tiled_step_scaling.py-236-            f"compiled tiled step contains full-cube all-gathers "
--
./scripts/bench/bench_cube_tiled_step_scaling.py-264-        from jax.experimental import multihost_utils
./scripts/bench/bench_cube_tiled_step_scaling.py-265-
./scripts/bench/bench_cube_tiled_step_scaling.py-266-        multihost_utils.sync_global_devices("cube_tiled_bench_start")
./scripts/bench/bench_cube_tiled_step_scaling.py-267-
./scripts/bench/bench_cube_tiled_step_scaling.py-268-    if args.closed_loop:
./scripts/bench/bench_cube_tiled_step_scaling.py:269:        # #921: the closed-loop step fuses the halo collective-permutes with
./scripts/bench/bench_cube_tiled_step_scaling.py-270-        # the in-stage mass-fixer psum in ONE executable; on multi-process GPU
./scripts/bench/bench_cube_tiled_step_scaling.py-271-        # the NCCL comm-init of those two clique kinds can be ordered
./scripts/bench/bench_cube_tiled_step_scaling.py-272-        # differently per rank and DEADLOCK.  Prime every clique in a fixed,
./scripts/bench/bench_cube_tiled_step_scaling.py-273-        # rank-independent order FIRST (no-op single-process / CPU-virtual).
./scripts/bench/bench_cube_tiled_step_scaling.py-274-        from legoesm.parallel.tiled_production_cdgrid import (
--
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-56-# under rec["metadata"] so a virtual-CPU-device proxy, a gloo/TCP fabric run,
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-57-# or an f32 ablation is falsifiable from the JSONL row alone.  metadata.py
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-58-# imports JAX lazily, so this is safe before jax.distributed.initialize.
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-59-from metadata import (  # noqa: E402
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-60-    annotate_incomplete,
./scripts/bench/bench_ocean_latlon_spmd_scaling.py:61:    calibrated_bound,
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-62-    comm_accounting,
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-63-    scaling_metadata,
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-64-    tidy_throughput_fields,
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-65-    wet_cell_metrics,
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-66-)
--
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-286-    p.add_argument("--fused-halo", action="store_true",
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-287-                   help="Opt-in SPMD halo message aggregation "
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-288-                        "(LEGOESM_LATLON_SPMD_FUSED_HALO=1): one ppermute "
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-289-                        "pair per direction per dtype group at every "
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-290-                        "pad_multi site instead of one per field — "
./scripts/bench/bench_ocean_latlon_spmd_scaling.py:291:                        "measured 25%% fewer static collective-permutes on "
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-292-                        "this step, bit-identical results. A/B against "
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-293-                        "the default run.")
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-294-    p.add_argument("--wide-halo", action="store_true",
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-295-                   help="Opt-in wide-halo split-explicit barotropic: one "
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-296-                        "fused wide lat-halo exchange per chunk of substeps "
--
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-605-
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-606-    # --- Communication accounting (audit item 4) + calibrated bound (8) ----
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-607-    # Analytic INTER-DEVICE census, barotropic-solver scope ONLY (the
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-608-    # baroclinic 3-D pads are not counted -> bytes/comm are a LOWER census,
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-609-    # flagged machine-readably via halo_bytes_is_lower_bound; T_bound is a
./scripts/bench/bench_ocean_latlon_spmd_scaling.py:610:    # heuristic model, see calibrated_bound's docstring).
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-611-    # implicit_cn PCG: each Helmholtz apply pads eta N+S (gradient stencil)
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-612-    # + the v-face flux row (divergence) ~= 2 exchanges/apply, applied
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-613-    # iters + 1 times (incl. the initial residual); reductions = the dot
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-614-    # batches (2/iter standard, 1/iter single_reduce) + the initial batch
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-615-    # + the mass-projection psum + the eta-floor clamp psum.
--
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-677-
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-678-    # Calibrated T_bound (audit item 8): nd=1 rows ARE their own compute
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-679-    # ingredient; nd>1 rows need the nd=1 fused number passed in (else the
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-680-    # bound is emitted null + flagged).  launch_host_ms=0: per-step launch
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-681-    # cost inside a fused lax.scan block is amortized to ~0.
./scripts/bench/bench_ocean_latlon_spmd_scaling.py:682:    bound_rec = calibrated_bound(
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-683-        measured_fused_step_ms=med,
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-684-        single_device_fused_step_ms=(med if nd == 1
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-685-                                     else args.single_dev_fused_ms),
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-686-        halo_messages_per_step=comm_rec["halo_messages_per_step"],
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-687-        halo_bytes_per_step=comm_rec["halo_bytes_per_step"],
--
./scripts/bench/run_levante_gpu_scaling.py-326-    # Collective-permute op census of the compiled TIMED executable
./scripts/bench/run_levante_gpu_scaling.py-327-    # (comm-minimisation step 1: measurement infrastructure for the
./scripts/bench/run_levante_gpu_scaling.py-328-    # upcoming halo-fusion work).  ``-1`` = not measured — single
./scripts/bench/run_levante_gpu_scaling.py-329-    # device, MPI, non-cubed-sphere, or the HLO guard was skipped via
./scripts/bench/run_levante_gpu_scaling.py-330-    # LEGOESM_SPMD_FORCE_ALLGATHER.  Sync ops lower as
./scripts/bench/run_levante_gpu_scaling.py:331:    # ``collective-permute``; async pairs as ``-start``/``-done``.
./scripts/bench/run_levante_gpu_scaling.py-332-    hlo_collective_permute: int = -1
./scripts/bench/run_levante_gpu_scaling.py-333-    hlo_collective_permute_start: int = -1
./scripts/bench/run_levante_gpu_scaling.py-334-    hlo_collective_permute_done: int = -1
./scripts/bench/run_levante_gpu_scaling.py-335-
./scripts/bench/run_levante_gpu_scaling.py-336-
--
./scripts/bench/run_levante_gpu_scaling.py-611-# ===========================================================================
./scripts/bench/run_levante_gpu_scaling.py-612-# Timed scan runner — shared between the dry-dycore and moist-segment paths
./scripts/bench/run_levante_gpu_scaling.py-613-# ===========================================================================
./scripts/bench/run_levante_gpu_scaling.py-614-
./scripts/bench/run_levante_gpu_scaling.py-615-def _count_collective_permute_ops(hlo_text: str) -> dict[str, int]:
./scripts/bench/run_levante_gpu_scaling.py:616:    """Census of collective-permute ops in a compiled HLO module.
./scripts/bench/run_levante_gpu_scaling.py-617-
./scripts/bench/run_levante_gpu_scaling.py-618-    Counts opcode *applications* (``<opcode>(``) so each op is counted
./scripts/bench/run_levante_gpu_scaling.py-619-    once regardless of how many times its result name appears.  Sync
./scripts/bench/run_levante_gpu_scaling.py:620:    halo exchanges lower to ``collective-permute``; the async form
./scripts/bench/run_levante_gpu_scaling.py:621:    lowers to ``collective-permute-start`` / ``collective-permute-done``
./scripts/bench/run_levante_gpu_scaling.py-622-    pairs.  Comm-minimisation sequencing step 1: this census is the
./scripts/bench/run_levante_gpu_scaling.py-623-    before/after metric for the upcoming halo-fusion work.
./scripts/bench/run_levante_gpu_scaling.py-624-    """
./scripts/bench/run_levante_gpu_scaling.py-625-    import re
./scripts/bench/run_levante_gpu_scaling.py-626-    return {
./scripts/bench/run_levante_gpu_scaling.py:627:        "collective-permute": len(
./scripts/bench/run_levante_gpu_scaling.py:628:            re.findall(r"\bcollective-permute\(", hlo_text)),
./scripts/bench/run_levante_gpu_scaling.py:629:        "collective-permute-start": len(
./scripts/bench/run_levante_gpu_scaling.py:630:            re.findall(r"\bcollective-permute-start\(", hlo_text)),
./scripts/bench/run_levante_gpu_scaling.py:631:        "collective-permute-done": len(
./scripts/bench/run_levante_gpu_scaling.py:632:            re.findall(r"\bcollective-permute-done\(", hlo_text)),
./scripts/bench/run_levante_gpu_scaling.py-633-    }
./scripts/bench/run_levante_gpu_scaling.py-634-
./scripts/bench/run_levante_gpu_scaling.py-635-
./scripts/bench/run_levante_gpu_scaling.py-636-def _hlo_census_fields(hlo_counts: dict[str, int] | None) -> dict[str, int]:
./scripts/bench/run_levante_gpu_scaling.py:637:    """``TimingResult`` kwargs for the collective-permute census.
./scripts/bench/run_levante_gpu_scaling.py-638-
./scripts/bench/run_levante_gpu_scaling.py-639-    Empty dict (→ the ``-1`` "not measured" defaults) when the HLO
./scripts/bench/run_levante_gpu_scaling.py-640-    guard did not run.
./scripts/bench/run_levante_gpu_scaling.py-641-    """
./scripts/bench/run_levante_gpu_scaling.py-642-    if hlo_counts is None:
./scripts/bench/run_levante_gpu_scaling.py-643-        return {}
./scripts/bench/run_levante_gpu_scaling.py-644-    return {
./scripts/bench/run_levante_gpu_scaling.py:645:        "hlo_collective_permute": hlo_counts["collective-permute"],
./scripts/bench/run_levante_gpu_scaling.py:646:        "hlo_collective_permute_start": hlo_counts["collective-permute-start"],
./scripts/bench/run_levante_gpu_scaling.py:647:        "hlo_collective_permute_done": hlo_counts["collective-permute-done"],
./scripts/bench/run_levante_gpu_scaling.py-648-    }
./scripts/bench/run_levante_gpu_scaling.py-649-
./scripts/bench/run_levante_gpu_scaling.py-650-
./scripts/bench/run_levante_gpu_scaling.py-651-def _build_timed_scan_runner(
./scripts/bench/run_levante_gpu_scaling.py-652-    *,
--
./scripts/bench/run_levante_gpu_scaling.py-679-       (plain ``jax.jit``) everywhere else — zero behavior change for
./scripts/bench/run_levante_gpu_scaling.py-680-       single-GPU / MPI / non-cubed-sphere rows;
./scripts/bench/run_levante_gpu_scaling.py-681-    2. sharding tripwire #1 on the post-warmup seed state;
./scripts/bench/run_levante_gpu_scaling.py-682-    3. the compiled-HLO hot-path guard: zero full-cube all-gathers in
./scripts/bench/run_levante_gpu_scaling.py-683-       the timed executable (LEGOESM_SPMD_FORCE_ALLGATHER=1 skips with
./scripts/bench/run_levante_gpu_scaling.py:684:       a loud warning), plus the collective-permute op census (printed
./scripts/bench/run_levante_gpu_scaling.py-685-       and returned for the result row metadata).  The
./scripts/bench/run_levante_gpu_scaling.py-686-       ``lower().compile()`` result is reused as the timed runner, so
./scripts/bench/run_levante_gpu_scaling.py-687-       the guard adds no extra compilation;
./scripts/bench/run_levante_gpu_scaling.py-688-    4. precompile against leaf-cloned state (so the timed run still
./scripts/bench/run_levante_gpu_scaling.py-689-       starts from the post-warmup state, and queued XLA work cannot
--
./scripts/bench/run_levante_gpu_scaling.py-801-                "ops in the timed executable",
./scripts/bench/run_levante_gpu_scaling.py-802-                flush=True,
./scripts/bench/run_levante_gpu_scaling.py-803-            )
./scripts/bench/run_levante_gpu_scaling.py-804-            hlo_counts = _count_collective_permute_ops(_hlo_text)
./scripts/bench/run_levante_gpu_scaling.py-805-            print(
./scripts/bench/run_levante_gpu_scaling.py:806:                f"    HLO census: {hlo_counts['collective-permute']} "
./scripts/bench/run_levante_gpu_scaling.py:807:                f"collective-permute, "
./scripts/bench/run_levante_gpu_scaling.py:808:                f"{hlo_counts['collective-permute-start']} -start, "
./scripts/bench/run_levante_gpu_scaling.py:809:                f"{hlo_counts['collective-permute-done']} -done op(s) "
./scripts/bench/run_levante_gpu_scaling.py-810-                f"in the timed executable",
./scripts/bench/run_levante_gpu_scaling.py-811-                flush=True,
./scripts/bench/run_levante_gpu_scaling.py-812-            )
./scripts/bench/run_levante_gpu_scaling.py-813-            scan_runner = _compiled_runner
./scripts/bench/run_levante_gpu_scaling.py-814-
--
./scripts/bench/metadata.py-98-def _env_flag_true(name: str) -> bool:
./scripts/bench/metadata.py-99-    """True iff env var ``name`` is a truthy flag ("1"/"true"/"yes"/"on")."""
./scripts/bench/metadata.py-100-    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")
./scripts/bench/metadata.py-101-
./scripts/bench/metadata.py-102-
./scripts/bench/metadata.py:103:# Match the OP-CALL form ``collective-permute(`` / ``collective_permute(`` /
./scripts/bench/metadata.py-104-# ``...-start(`` (a paren directly after the op name), NOT bare substrings: the
./scripts/bench/metadata.py-105-# COMPILED-HLO config header echoes XLA_FLAGS, so a flag name like
./scripts/bench/metadata.py-106-# ``xla_gpu_collective_permute_combine_threshold_bytes=`` (set by #1175's
./scripts/bench/metadata.py-107-# MPAS_CP_COMBINE) would false-match a plain substring scan and over-count.
./scripts/bench/metadata.py-108-_COLLECTIVE_PERMUTE_RE = re.compile(r"collective[_-]permute(?:[_-]start)?\(")
--
./scripts/bench/metadata.py-135-    ``hlo_collective_permutes`` (cube tiled, MPAS ico) — no re-implementation."""
./scripts/bench/metadata.py-136-    return _count_op_calls(hlo_text, _COLLECTIVE_PERMUTE_RE)
./scripts/bench/metadata.py-137-
./scripts/bench/metadata.py-138-
./scripts/bench/metadata.py-139-def hlo_collective_permutes(fn, *args) -> int | None:
./scripts/bench/metadata.py:140:    """Best-effort: count the collective-permutes in the COMPILED HLO of
./scripts/bench/metadata.py-141-    ``fn(*args)``.
./scripts/bench/metadata.py-142-
./scripts/bench/metadata.py-143-    Compiles (``.lower(...).compile().as_text()``), NOT bare
./scripts/bench/metadata.py-144-    ``.lower().as_text()``: the census must reflect the EXECUTABLE's round
./scripts/bench/metadata.py:145:    count, because XLA collective-permute combining / pipelined-p2p
./scripts/bench/metadata.py-146-    (#1175, ``MPAS_CP_COMBINE``) fuses rounds during optimization — the whole
./scripts/bench/metadata.py-147-    metric #1113 tracks. Pre-optimization StableHLO would overstate CPs versus
./scripts/bench/metadata.py-148-    the timed executable. Matches the cube tiled bench, which compiles too.
./scripts/bench/metadata.py-149-    Returns ``None`` (never raises) if lowering/compilation is unsupported OR
./scripts/bench/metadata.py-150-    the backend's ``as_text()`` yields no HLO, so a timing probe can record
--
./scripts/bench/metadata.py-167-    inflate the count (same guard as :data:`_COLLECTIVE_PERMUTE_RE`)."""
./scripts/bench/metadata.py-168-    stem = op_name.replace("-", "[_-]")
./scripts/bench/metadata.py-169-    return re.compile(stem + r"(?:[_-]start)?\(")
./scripts/bench/metadata.py-170-
./scripts/bench/metadata.py-171-
./scripts/bench/metadata.py:172:#: Every collective OP family a scaling row can run.  ``collective-permute`` is
./scripts/bench/metadata.py-173-#: the band/face halo (reuse the canonical permute regex so its count stays
./scripts/bench/metadata.py-174-#: bit-identical to :func:`count_collective_permutes`); ``all-reduce`` is the
./scripts/bench/metadata.py-175-#: conservation fixer AND the ocean implicit-CN PCG reduction wall (~120/step —
./scripts/bench/metadata.py-176-#: the #1 ocean strong-scaling bottleneck, invisible to a permute-only census);
./scripts/bench/metadata.py-177-#: the rest surface any SPMD resharding an operator introduces.
--
./scripts/bench/metadata.py-621-        "sypd": sypd,
./scripts/bench/metadata.py-622-        "mcells_per_s": mcells_per_s,
./scripts/bench/metadata.py-623-    }
./scripts/bench/metadata.py-624-
./scripts/bench/metadata.py-625-
./scripts/bench/metadata.py:626:#: Placeholder comm-fabric numbers for :func:`calibrated_bound` when the
./scripts/bench/metadata.py-627-#: caller passes no measured values.  Ballpark single-node GPU-interconnect
./scripts/bench/metadata.py-628-#: figures (order NVLink/PCIe), NOT measurements of THIS machine —
./scripts/bench/metadata.py-629-#: MACHINE-CALIBRATED-REQUIRED: any bound built on them is emitted with
./scripts/bench/metadata.py-630-#: ``bound_calibrated=False`` and must never be quoted as a hardware
./scripts/bench/metadata.py-631-#: roofline.  Calibrate with a ping-pong / allreduce microbenchmark on the
--
./scripts/bench/metadata.py-749-        out["wet_cell_levels_per_device_min"] = min(per)
./scripts/bench/metadata.py-750-        out["wet_cell_levels_per_device_max"] = max(per)
./scripts/bench/metadata.py-751-    return out
./scripts/bench/metadata.py-752-
./scripts/bench/metadata.py-753-
./scripts/bench/metadata.py:754:def calibrated_bound(
./scripts/bench/metadata.py-755-    *,
./scripts/bench/metadata.py-756-    measured_fused_step_ms: float | None = None,
./scripts/bench/metadata.py-757-    single_device_fused_step_ms: float | None = None,
./scripts/bench/metadata.py-758-    halo_messages_per_step: int | None = None,
./scripts/bench/metadata.py-759-    halo_bytes_per_step: int | None = None,
--
./scripts/bench/metadata.py-811-    lat_us = (DEFAULT_COMM_LATENCY_US if latency_us is None
./scripts/bench/metadata.py-812-              else float(latency_us))
./scripts/bench/metadata.py-813-    bw_gbs = (DEFAULT_COMM_BANDWIDTH_GBS if bandwidth_GBs is None
./scripts/bench/metadata.py-814-              else float(bandwidth_GBs))
./scripts/bench/metadata.py-815-    if lat_us < 0.0:
./scripts/bench/metadata.py:816:        raise ValueError(f"calibrated_bound: latency_us must be >= 0, "
./scripts/bench/metadata.py-817-                         f"got {lat_us}")
./scripts/bench/metadata.py-818-    if bw_gbs <= 0.0:
./scripts/bench/metadata.py:819:        raise ValueError(f"calibrated_bound: bandwidth_GBs must be > 0, "
./scripts/bench/metadata.py-820-                         f"got {bw_gbs}")
./scripts/bench/metadata.py-821-
./scripts/bench/metadata.py-822-    missing = [name for name, v in (
./scripts/bench/metadata.py-823-        ("single_device_fused_step_ms", single_device_fused_step_ms),
./scripts/bench/metadata.py-824-        ("halo_messages_per_step", halo_messages_per_step),
--
./docs/science/specs/FV3_3D.md-1924-very_long_time).  Combined with the iter-43 ``LEGOESM_AH_SCALE``
./docs/science/specs/FV3_3D.md-1925-auto-apply (1.0 / 2.0 / 10.0 by resolution bucket), this is a
./docs/science/specs/FV3_3D.md-1926-single env var pair recommended for cube HS.
./docs/science/specs/FV3_3D.md-1927-
./docs/science/specs/FV3_3D.md-1928-C96 30-day empirical validation of the iter-81 dt=50 setting is
./docs/science/specs/FV3_3D.md:1929:**EMPIRICALLY CONFIRMED** as of iter 99: full 30-day run completed
./docs/science/specs/FV3_3D.md-1930-finite with max|u|=20.14 m/s, max|v|=11.84 m/s, 51840 steps, 1755 s
./docs/science/specs/FV3_3D.md-1931-wall.  iter-85's linear-in-1/dt eigenmode prediction validated
./docs/science/specs/FV3_3D.md-1932-(predicted NaN at day 45; run stopped finite at day 30).
./docs/science/specs/FV3_3D.md-1933-
./docs/science/specs/FV3_3D.md-1934-The full iteration log follows.
--
./docs/science/specs/FV3_3D.md-1952-    diffusive CFL ``A_h × dt / dx²`` exceeds 0.5.  The right
./docs/science/specs/FV3_3D.md-1953-    response to instability is sometimes SMALLER ``dt``, not
./docs/science/specs/FV3_3D.md-1954-    larger ``A_h``.
./docs/science/specs/FV3_3D.md-1955-
./docs/science/specs/FV3_3D.md-1956-3.  **Smaller ``dt`` DELAYS the eigenmode in proportion to 1/dt
./docs/science/specs/FV3_3D.md:1957:    (iter-95 EMPIRICALLY CONFIRMED).**  iter-69 (``dt=150``: NaN
./docs/science/specs/FV3_3D.md-1958-    day 15), iter-79 (``dt=100``: NaN day 22.5).  Note
./docs/science/specs/FV3_3D.md-1959-    ``15 × 1.5 = 22.5`` — exactly proportional.  iter-85
./docs/science/specs/FV3_3D.md-1960-    extrapolated: ``dt=50`` should NaN at ~day 45 (survives 30 d).
./docs/science/specs/FV3_3D.md-1961-    iter-95 EMPIRICALLY VALIDATED this: C96 dt=50 ran past day
./docs/science/specs/FV3_3D.md-1962-    22.5 (max|u|=14.04 m/s at step 38880), proving the eigenmode
--
./scripts/bench/bench_ocean_latlon_spmd_pcg.py-11-mpi4jax — runs on route-B (cuda-jax, the RTX8000 PCIe pair) where mpi4jax
./scripts/bench/bench_ocean_latlon_spmd_pcg.py-12-is unavailable.
./scripts/bench/bench_ocean_latlon_spmd_pcg.py-13-
./scripts/bench/bench_ocean_latlon_spmd_pcg.py-14-STRONG scaling: a FIXED global grid solved on 1 device (whole grid) then N
./scripts/bench/bench_ocean_latlon_spmd_pcg.py-15-devices (each a latitude band).  ``efficiency(N) = t(1) / (N * t(N))``;
./scripts/bench/bench_ocean_latlon_spmd_pcg.py:16:ideal 1.0.  On a PCIe pair (no NVLink) the psum collective-permute is the
./scripts/bench/bench_ocean_latlon_spmd_pcg.py-17-expected ceiling — this bench MEASURES that ceiling.
./scripts/bench/bench_ocean_latlon_spmd_pcg.py-18-
./scripts/bench/bench_ocean_latlon_spmd_pcg.py-19-The operator is the uniform-coefficient 5-point Helmholtz ``A = I +
./scripts/bench/bench_ocean_latlon_spmd_pcg.py-20-c(-Delta)`` (built on the backend-oblivious ``pad_halo_latlon``, so the
./scripts/bench/bench_ocean_latlon_spmd_pcg.py-21-SAME closure runs serial and sharded).  Its COMMUNICATION pattern (band
--
./scripts/cluster/scaling_levante/README.md-72-| File | What |
./scripts/cluster/scaling_levante/README.md-73-|---|---|
./scripts/cluster/scaling_levante/README.md-74-| `gpu_scaling.sbatch` | ATM (cube AMIP physics, single-process multi-GPU) + OCEAN (`bench_ocean_latlon_spmd_scaling.py`, 1/2/4 A100 with parity+conservation smoke) on one GPU node. Plain GPU env — no mpi4jax. |
./scripts/cluster/scaling_levante/README.md-75-| `gpu_multinode_scaling.sbatch` | MULTI-NODE route-B lanes over NCCL/IB (SLURM auto-detected `jax.distributed`): A = cube `--cs-spmd` (6 GPU / 2 nodes), C = atm lat-lon `--multicontroller` (8 GPU), D = ocean `--multicontroller` (8 GPU). |
./scripts/cluster/scaling_levante/README.md-76-| `cpu_scaling.sbatch` | ATM + OCEAN CPU-MPI rank ladders on `compute` nodes (`legoesm-mpi` env), with fail-fast smokes. |
./scripts/cluster/scaling_levante/README.md:77:| `diagnosis.sbatch` | BOTTLENECK diagnosis via `scripts/bench/run_scaling_diagnosis.py` — per-phase halo bandwidth, reduction latency, roofline, compute/comm overlap, and the **static collective census** (collective-permute + all-reduce + all-gather per step) the throughput jobs do NOT capture. Climbs the cube face-shard `1 2 3` ladder so the message-count-vs-shard curve — the LATENCY-bound anti-scaling signal — is recorded. `sbatch --export=ALL,MODE=census` for counts only. Derecho twin: `scaling_derecho/diagnosis.pbs`. |
./scripts/cluster/scaling_levante/README.md-78-
./scripts/cluster/scaling_levante/README.md-79-`_env.sh` now also carries the NCCL-over-IB defaults for the route-B lanes
./scripts/cluster/scaling_levante/README.md-80-(`NCCL_IB_HCA=mlx5`, `NCCL_SOCKET_IFNAME=ib0`, `NCCL_NET_GDR_LEVEL=PHB`,
./scripts/cluster/scaling_levante/README.md-81-`NCCL_CROSS_NIC=1`) — independent of, and coexisting with, the UCX/MPI
./scripts/cluster/scaling_levante/README.md-82-settings used by route-A. First multi-node run: `NCCL_DEBUG=INFO` must show
--
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-17-# same steps/warmup as the s9 ladder.  Weak pairs at matched cells/GPU
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-18-# (81.9k / 41.0k / 20.5k) are computed ONLY from these rows vs 26600095.
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-19-#
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-20-# Falsifiability, written BEFORE submit:
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-21-#   numbers : s8-lloyd0 np8/16/32 steady_median_ms
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:22:#   CONFIRM (a matched-tile scale-out term exists): s9/s8 ratios at
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-23-#             matched cells/GPU stay well above 1 (prior draft saw
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-24-#             1.80/1.35/1.41 on the CONFOUNDED pairs)
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:25:#   REFUTE  : ratios collapse toward ~1.0 -> the draft's "term" was the
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-26-#             Lloyd-mesh confound, and MPAS-GPU weak scaling is near
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-27-#             ideal at matched tile.
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-28-# Mesh: prewarmed into LEGOESM_MESH_CACHE_DIR (subdiv-8 is below the
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-29-# big-mesh refuse threshold, so a cache miss falls back to in-process
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-30-# builds — slower, still correct).
--
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-223-
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-224-# Lane T (RUN_TUNE=1): comm-tuning A/B ladder — Derecho twin (see
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-225-# gpu_multinode_scaling.pbs lane T + docs/performance/scaling/
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-226-# spmd_message_census_2026-07-08.md). Arms: base (control, same allocation),
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-227-# fused (LEGOESM_LATLON_SPMD_FUSED_HALO=1 — bit-identical multi-pad packing,
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:228:# trace receipt atm 41->29 CPs/step), xla (collective-permute combining +
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-229-# pipelined p2p), pgle (profile-guided latency estimation; recompiles after
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-230-# the profiling runs — bench-jit-safe, not for AOT jobs). Outputs land under
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-231-# _ab_tuning/ which aggregate_bcw_scaling SKIPS (A/B receipts never join the
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-232-# scaling curves); compare arms via steady_median_ms / sypd per JSONL row.
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-233-RUN_TUNE="${RUN_TUNE:-0}"
--
./scripts/cluster/scaling_levante/diagnosis.sbatch-12-# BOTTLENECK DIAGNOSIS on a Levante GPU node (4x A100-80GB, NVLink).
./scripts/cluster/scaling_levante/diagnosis.sbatch-13-# Levante twin of scaling_derecho/diagnosis.pbs — identical tool + ladder.
./scripts/cluster/scaling_levante/diagnosis.sbatch-14-#
./scripts/cluster/scaling_levante/diagnosis.sbatch-15-# Runs scripts/bench/run_scaling_diagnosis.py — the per-phase bottleneck tool
./scripts/cluster/scaling_levante/diagnosis.sbatch-16-# (halo bandwidth, reduction latency, roofline, compute/comm overlap, and the
./scripts/cluster/scaling_levante/diagnosis.sbatch:17:# STATIC collective census: collective-permute + all-reduce + all-gather per
./scripts/cluster/scaling_levante/diagnosis.sbatch-18-# step) — which the throughput jobs (gpu_scaling.sbatch) do NOT capture.
./scripts/cluster/scaling_levante/diagnosis.sbatch-19-#
./scripts/cluster/scaling_levante/diagnosis.sbatch-20-# Why this matters: the measured f64==f32 GPU strong-scaling curves mean the
./scripts/cluster/scaling_levante/diagnosis.sbatch-21-# multi-GPU leg is LATENCY-bound, so the per-step MESSAGE COUNT — captured by
./scripts/cluster/scaling_levante/diagnosis.sbatch-22-# the census below — is the lever, not the byte volume.  Cube face-sharding
--
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-24-#   solo_post : solo again AFTER phase B (brackets ordering/thermal
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-25-#               drift; contrast uses mean of the two solos)
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-26-#
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-27-# Falsifiability, written BEFORE submit:
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-28-#   numbers : 2 solo + 4 replica steady_median_ms
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:29:#   CONFIRM : max(replica) <= 1.10 x mean(solo) -> guaranteed aggregate
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-30-#             >= 4/1.10 = 3.64x the 32-GPU solo rate (~19.9 GC/s if solo
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-31-#             reproduces 5.47) = ~3.3x the observed 128-GPU
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-32-#             single-trajectory rate.  NOT "4x": 1.10 is the bar, the
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-33-#             margin below it is the measured contention.
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:34:#   REFUTE  : any replica > 1.10x solo -> contention term, quantified
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-35-#             per replica.
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-36-# Protocol: config identical to job 26600095 np32 rung (sfc, lloyd 0,
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-37-# f32, padded-128 reorder) EXCEPT steps 5000 / warmup 100 so the stepping
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-38-# window (~60 s at 12.5 ms/step) dwarfs launch skew between replicas --
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-39-# overlap is EVIDENCED, not assumed, by the per-step Start/End + NodeList
--
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-10-#SBATCH --time=01:30:00
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-11-#SBATCH --output=ll128_comb.%j.log
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-12-# CP-COMBINING A/B at LL2048@128 (the measured/bound ~2.9 gap).
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-13-# The calibrated bound (census 41 CP + 1 AR/step, exact bytes 18.5 MB/dev,
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-14-# IB 26.3us/23.5GB/s, nd=1 same-tile compute 1.659 ms) models 1.85-1.89 ms;
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:15:# measured is 5.58 (job 26534060). Leading PLAUSIBLE mechanism: effective
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-16-# per-CP overhead (launch+schedule+sync) >> raw fabric latency across 41
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-17-# dependency-chained exchanges — the arXiv:2607.16100 regime. Lever:
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-18-# COMBINE independent CPs into fewer, larger messages.
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-19-# NOTE: the closed-levers null for these flags was the OCEAN lane
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-20-# (reduction-dominated); this is the first atm-latlon test — not a rerun
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-21-# of a closed null.
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-22-# Falsifiability, BEFORE submit — arms byte-matched to 26534060 protocol:
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-23-#   A control (default flags)      : expect ~5.6 ms
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:24:#   B +cp-combine 8MB threshold    : CONFIRM lever if >=10% under A
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-25-#   C +combine +pipelined-p2p      : scheduling interaction
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:26:#   REFUTE if B,C within 2% of A -> overhead is not combinable-CP count;
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-27-#   next hypothesis = unoverlapped serial chain (scheduling lever).
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-28-set -uo pipefail
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-29-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-30-export JAX_PLATFORMS=cuda,cpu
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-31-export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
--
./scripts/cluster/scaling_derecho/README.md-483-|---|---|
./scripts/cluster/scaling_derecho/README.md-484-| `ocean_gpu_scaling.pbs` | OCEAN weak+strong on one GPU node via `scripts/bench/bench_ocean_latlon_spmd_scaling.py` (full lat-lon C-grid step sharded over 1/2/4 A100; fail-fast `--parity-gate` + `--check-conservation` smoke first). Plain GPU env — no mpi4jax. |
./scripts/cluster/scaling_derecho/README.md-485-| `ocean_cpu_scaling.pbs` | OCEAN weak+strong CPU-MPI rank ladder (`bench_ocean_mpi_scaling.py`, `legoesm-mpi` env), with a 2-rank parity+conservation smoke. |
./scripts/cluster/scaling_derecho/README.md-486-| `gpu_multinode_scaling.pbs` | MULTI-NODE GPU lanes over jax.distributed + NCCL: A = cube `--cs-spmd` (6 GPU / 2 nodes), C = atm lat-lon `--multicontroller` (8 GPU), D = ocean `--multicontroller` (8 GPU); plus the optional route-A CUDA-aware mpi4jax lane (`RUN_ROUTEA=1`, needs the overlay env) and the comm-tuning A/B ladder (`RUN_TUNE=1`, lane T below). |
./scripts/cluster/scaling_derecho/README.md-487-| `build_nccl_ofi.sh` | Login-node build of **aws-ofi-nccl** against Derecho's Cray libfabric (no NCCL build dep — the plugin vendors the net-API headers and is dlopen'd by the jax-wheel NCCL). |
./scripts/cluster/scaling_derecho/README.md:488:| `diagnosis.pbs` | BOTTLENECK diagnosis via `scripts/bench/run_scaling_diagnosis.py` — per-phase halo bandwidth, reduction latency, roofline, compute/comm overlap, and the **static collective census** (collective-permute + all-reduce + all-gather per step) that the throughput jobs above do NOT capture. Climbs the cube face-shard `1 2 3` ladder (must divide 6) so the message-count-vs-shard curve — the LATENCY-bound anti-scaling signal — is recorded. `qsub -v MODE=census` for counts only. |
./scripts/cluster/scaling_derecho/README.md-489-
./scripts/cluster/scaling_derecho/README.md-490-NCCL on Slingshot-11 has NO native CXI support: without the plugin the
./scripts/cluster/scaling_derecho/README.md-491-multi-node lanes fall back to TCP sockets over `hsn` (correct, 2-3x slower
./scripts/cluster/scaling_derecho/README.md-492-comm — loud warning, fine for shakeout). For production numbers:
./scripts/cluster/scaling_derecho/README.md-493-
--
./scripts/cluster/scaling_derecho/README.md-520-
./scripts/cluster/scaling_derecho/README.md-521-## 2026-07 lane T: comm-tuning A/B ladder (`RUN_TUNE=1`)
./scripts/cluster/scaling_derecho/README.md-522-
./scripts/cluster/scaling_derecho/README.md-523-Once the route-B lanes are green on this machine, the remaining strong-
./scripts/cluster/scaling_derecho/README.md-524-scaling headroom at small tiles is **per-step message count × latency**
./scripts/cluster/scaling_derecho/README.md:525:(census: cube 46 collective-permutes/step at 6 devices with field packing
./scripts/cluster/scaling_derecho/README.md-526-already at floor; atm latlon 41 → 29 behind the fused-halo flag — see
./scripts/cluster/scaling_derecho/README.md-527-`docs/performance/scaling/spmd_message_census_2026-07-08.md`). Lane T runs
./scripts/cluster/scaling_derecho/README.md-528-the ranked rungs as same-allocation A/B arms (a fresh `base` control arm is
./scripts/cluster/scaling_derecho/README.md-529-re-run in the same job — never compare against an earlier job's numbers):
./scripts/cluster/scaling_derecho/README.md-530-
--
./scripts/cluster/scaling_derecho/diagnosis.pbs-10-# ===========================================================================
./scripts/cluster/scaling_derecho/diagnosis.pbs-11-# BOTTLENECK DIAGNOSIS on a Derecho GPU node (4x A100-40GB, NVLink).
./scripts/cluster/scaling_derecho/diagnosis.pbs-12-#
./scripts/cluster/scaling_derecho/diagnosis.pbs-13-# Runs scripts/bench/run_scaling_diagnosis.py — the per-phase bottleneck tool
./scripts/cluster/scaling_derecho/diagnosis.pbs-14-# (halo bandwidth, reduction latency, roofline, compute/comm overlap, and the
./scripts/cluster/scaling_derecho/diagnosis.pbs:15:# STATIC collective census: collective-permute + all-reduce + all-gather per
./scripts/cluster/scaling_derecho/diagnosis.pbs-16-# step) — which the throughput jobs (scaling_gpu.sh, ocean_gpu_scaling.pbs)
./scripts/cluster/scaling_derecho/diagnosis.pbs-17-# do NOT capture.  This is the lane that answers "WHERE is the strong-scaling
./scripts/cluster/scaling_derecho/diagnosis.pbs-18-# loss", not just "what is the SYPD".
./scripts/cluster/scaling_derecho/diagnosis.pbs-19-#
./scripts/cluster/scaling_derecho/diagnosis.pbs-20-# Why this matters (measured campaign readings, derecho_levante_sota_review):
./scripts/cluster/scaling_derecho/diagnosis.pbs-21-#   * f64 and f32 GPU strong-scaling curves COINCIDE => the multi-GPU leg is
./scripts/cluster/scaling_derecho/diagnosis.pbs-22-#     LATENCY-bound, not bandwidth-bound.  The lever is the per-step MESSAGE
./scripts/cluster/scaling_derecho/diagnosis.pbs-23-#     COUNT, not the byte volume — so the collective census below is the
./scripts/cluster/scaling_derecho/diagnosis.pbs-24-#     first-class signal, and it is STATIC (a CPU compile gives the same count,
./scripts/cluster/scaling_derecho/diagnosis.pbs-25-#     but we capture it here on the real executable alongside the timings).
./scripts/cluster/scaling_derecho/diagnosis.pbs:26:#   * Cube face-shard anti-scales at small tiles because the collective-permute
./scripts/cluster/scaling_derecho/diagnosis.pbs-27-#     count grows with the shard count (~12 at 2 dev, ~41 at 6 dev on C24/L8)
./scripts/cluster/scaling_derecho/diagnosis.pbs-28-#     while the per-message NCCL p2p latency (~30-80 us) does not amortise.
./scripts/cluster/scaling_derecho/diagnosis.pbs-29-#
./scripts/cluster/scaling_derecho/diagnosis.pbs-30-# Cube face-sharding requires the device count to DIVIDE 6 (1/2/3/6); a 4-GPU
./scripts/cluster/scaling_derecho/diagnosis.pbs-31-# node therefore climbs the 1/2/3 ladder (4 is not face-divisible, and >6 uses
--
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-36-#   Lane A: plain `legoesm-gpu` (README Step 1) is enough.
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-37-#   Lane B: `legoesm-gpu` WITH the route-A overlay (README Step 1b).
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-38-# The job runs lane A by default; enable lanes via qsub -v:
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-39-#   RUN_NCCL=1 (default 1)   RUN_ROUTEA=1 (default 0 — needs overlay)
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-40-#   RUN_TUNE=1 (default 0)   — lane T comm-tuning A/B ladder (fused-halo /
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:41:#     XLA collective-permute combining + pipelined p2p / PGLE) on the latlon
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-42-#     + cube lanes; outputs under _ab_tuning/ (excluded from the curves).
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-43-#     Run it AFTER the main lanes are green on this machine; see
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-44-#     docs/performance/scaling/spmd_message_census_2026-07-08.md.
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-45-#
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-46-# SUBMIT:
--
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-84-NLEV="${NLEV:-26}"
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-85-OC_NLEV="${OC_NLEV:-20}"
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-86-N_WARMUP="${N_WARMUP:-3}"
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-87-N_TIMING="${N_TIMING:-30}"
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-88-
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:89:# XLA collective-permute combining + pipelined p2p. #1113 found the route-B MPAS
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-90-# (lane E) throughput wall is the ppermute ROUND COUNT (9->24->30 as the SFC
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-91-# partition gains edge-coloring rounds) times a fixed ~0.11 ms/round launch
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-92-# floor — NOT transport bandwidth (GDR-off was immaterial at these message
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:93:# sizes). These flags fuse adjacent collective-permutes and pipeline the p2p,
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-94-# attacking the round count directly (the issue's top-priority remedy). Shared
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-95-# with the lane-T A/B `xla` arm. On by default for lane E; MPAS_CP_COMBINE=0
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-96-# for a baseline arm.
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-97-_XLA_COMM_FLAGS="--xla_gpu_collective_permute_combine_threshold_bytes=33554432 --xla_gpu_enable_pipelined_p2p=true"
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-98-MPAS_CP_COMBINE="${MPAS_CP_COMBINE:-1}"  # lane E CP-combining default (#1113)
--
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-284-#   base  : lane-C/D env exactly as above (the control arm — re-run so every
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-285-#           arm shares one allocation/fabric state; NEVER compare to an
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-286-#           earlier job's numbers)
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-287-#   fused : LEGOESM_LATLON_SPMD_FUSED_HALO=1 (bit-identical multi-pad packing;
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-288-#           trace-level receipt: atm 41->29 CPs/step, ocean 25% fewer)
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:289:#   xla   : collective-permute combining + pipelined p2p — the cube/latlon
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-290-#           rounds are data-independent, so the GPU combiner can merge what
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-291-#           the SPMD partitioner emits separately (code round floor = 4)
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-292-#   pgle  : profile-guided latency estimation (recompiles after profiling
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-293-#           runs — bench-jit-safe, NOT for AOT jobs; that is why it is not a
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-294-#           backend.py default)
--
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-14-# Stage A (single process, CPU virtual devices): the tiled-vs-serial PARITY
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-15-#   gate at the adapter tolerances — correctness receipt BEFORE any timing.
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-16-#   Virtual-CPU rows are auto-flagged in metadata and are NOT speedup rows.
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-17-# Stage B (multicontroller, one process per GPU, route-B NCCL): the timed
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-18-#   run.  The bench REFUSES to record a row whose compiled HLO shows a
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs:19:#   full-cube all-gather (replicated execution) or zero collective-permutes.
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-20-#
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-21-# The tiled step is the DYNAMICS-ONLY base cut (the adapter refuses configs
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-22-# with hyperdiff/div-damp/sponge-implicit/duogrid — see
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-23-# tiled_step_adapter._refuse).  Compare against the face-only 6-GPU lane
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-24-# (blocker1_cube_shardmap.pbs / run_levante --cs-spmd) for the <=6 leg.
--
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-75-export NCCL_SOCKET_IFNAME=hsn
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-76-# #921: the closed-loop generation adds ONE extra collective the single-shot
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-77-# lane lacks — the dry-mass fixer's GLOBAL all-reduce
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-78-# (tiled_production_cdgrid._tile_fix_ps_mass_delta:
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-79-# psum over ("face","tile_i","tile_j")) layered on top of the halo
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs:80:# collective-permutes (cubesphere_exchange ppermute over the SAME axes). Under
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-81-# XLA GPU defaults (latency-hiding scheduler ON, async collectives ON,
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-82-# nccl_comm_splitting ON) the all-reduce clique and the permute cliques init
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-83-# CONCURRENTLY and the background NCCL init thread can establish them in a
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-84-# different relative order per rank -> cyclic wait / comm-init deadlock (ranks
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs-85-# reach channel/P2P setup but log ZERO "Init COMPLETE"). The code fix
--
./scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-20-PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
./scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-21-echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
./scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-22-echo "=== worktree ==="; git -C $WT log --oneline -1
./scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-23-
./scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-24-run_acc () {  # $1 = bottom_drag_scheme
./scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch:25:  JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 GM_CONFIRM=0 $PY - "$1" <<'PYEOF' 2>&1 | grep -vE "FutureWarning|warnings.warn|XLA|cuda|CUDA|Plugin"
./scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-26-import sys, dataclasses, numpy as np, jax
./scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-27-jax.config.update("jax_enable_x64", True)
./scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-28-scheme = sys.argv[1]
./scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-29-from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
./scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch-30-from legoesm.ocean.experiments.dino import (DINOConfig, create_dino_z_star,
--
./packages/core/legoesm/grids/topography.py-28-# ocean depth exceeds ~11000 m (Mariana), but the GEOPOTENTIAL of Everest is
./packages/core/legoesm/grids/topography.py-29-# ~86800 m²/s².  An "elevation" field whose magnitude exceeds this threshold is
./packages/core/legoesm/grids/topography.py-30-# almost certainly geopotential [m²/s²] mistakenly passed as elevation [m] —
./packages/core/legoesm/grids/topography.py-31-# multiplying it by g again in load_real_topography would silently inflate
./packages/core/legoesm/grids/topography.py-32-# phis ~9.8×.
./packages/core/legoesm/grids/topography.py:33:_MAX_PLAUSIBLE_ELEV_M = 12000.0
./packages/core/legoesm/grids/topography.py-34-
./packages/core/legoesm/grids/topography.py-35-
./packages/core/legoesm/grids/topography.py-36-def _grid_lat_lon_2d(grid):
./packages/core/legoesm/grids/topography.py-37-    """Per-cell (lat, lon) in the grid's native horizontal layout [rad].
./packages/core/legoesm/grids/topography.py-38-
--
./packages/core/legoesm/grids/topography.py-1176-
./packages/core/legoesm/grids/topography.py-1177-    # --- geopotential-vs-meters "double-g" guard -----------------------------
./packages/core/legoesm/grids/topography.py-1178-    # This loader multiplies elevation [m] by constants.g below; an ERA5-style
./packages/core/legoesm/grids/topography.py-1179-    # invariant where 'z' is GEOPOTENTIAL [m²/s²] would be silently inflated
./packages/core/legoesm/grids/topography.py-1180-    # ~9.8×.  Detect it by the units attribute (when present) and by magnitude
./packages/core/legoesm/grids/topography.py:1181:    # (see _MAX_PLAUSIBLE_ELEV_M).
./packages/core/legoesm/grids/topography.py-1182-    _double_g_msg = (
./packages/core/legoesm/grids/topography.py-1183-        f"Topography variable {elev_var!r} in {path!r} looks like surface "
./packages/core/legoesm/grids/topography.py-1184-        f"GEOPOTENTIAL [m**2 s**-2], not elevation [m] "
./packages/core/legoesm/grids/topography.py-1185-        f"(units={elev_units!r}, max |value| = "
./packages/core/legoesm/grids/topography.py-1186-        f"{float(np.max(np.abs(elev_data))):.0f}; no Earth elevation exceeds "
./packages/core/legoesm/grids/topography.py:1187:        f"~8850 m, threshold {_MAX_PLAUSIBLE_ELEV_M:.0f} m). Multiplying it "
./packages/core/legoesm/grids/topography.py-1188-        "by g again (the double-g trap) would inflate phis ~9.8x. Divide the "
./packages/core/legoesm/grids/topography.py-1189-        "field by g (legoesm.constants.g) first, or pass the correct "
./packages/core/legoesm/grids/topography.py-1190-        "elevation variable via config.elev_var."
./packages/core/legoesm/grids/topography.py-1191-    )
./packages/core/legoesm/grids/topography.py-1192-    _units_norm = elev_units.replace(" ", "").replace("**", "^").lower()
./packages/core/legoesm/grids/topography.py-1193-    if _units_norm in ("m^2s^-2", "m^2/s^2", "m2s-2", "m2/s2", "m^2s-2"):
./packages/core/legoesm/grids/topography.py-1194-        raise ValueError(_double_g_msg)
./packages/core/legoesm/grids/topography.py:1195:    if float(np.max(np.abs(elev_data))) > _MAX_PLAUSIBLE_ELEV_M:
./packages/core/legoesm/grids/topography.py-1196-        raise ValueError(_double_g_msg)
./packages/core/legoesm/grids/topography.py-1197-
./packages/core/legoesm/grids/topography.py-1198-    # Use protocol for grid detection (classification fixed in
./packages/core/legoesm/grids/topography.py-1199-    # _target_grid_degrees: by coordinate rank, so the lat-lon grid is no longer
./packages/core/legoesm/grids/topography.py-1200-    # mis-routed into the cubed-sphere smoother).
--
./scripts/matrix/run_atmosphere_test_matrix.py-904-this NaNs at day 22.5 — STILL INSUFFICIENT for 30-day."""
./scripts/matrix/run_atmosphere_test_matrix.py-905-
./scripts/matrix/run_atmosphere_test_matrix.py-906-_CFL_SAFETY_VERY_LONG_TIME: float = 0.154
./scripts/matrix/run_atmosphere_test_matrix.py-907-"""iter 80 calibration: dt=50 at C96.  iter-79 found dt=100 NaN's
./scripts/matrix/run_atmosphere_test_matrix.py-908-at day 22.5; this halves dt further as the next attempt at 30-day
./scripts/matrix/run_atmosphere_test_matrix.py:909:stability.  iter 99 EMPIRICALLY CONFIRMED full 30-day finite at
./scripts/matrix/run_atmosphere_test_matrix.py-910-C96 (max|u|=20.14 m/s, max|v|=11.84 m/s, 51840 steps, 1755 s
./scripts/matrix/run_atmosphere_test_matrix.py-911-wall).  iter-85 linear-in-1/dt prediction held: dt=50 was
./scripts/matrix/run_atmosphere_test_matrix.py-912-predicted to NaN at day 45; never reached because the 30-day run
./scripts/matrix/run_atmosphere_test_matrix.py-913-completed finite at day 30."""
./scripts/matrix/run_atmosphere_test_matrix.py-914-
--
./scripts/matrix/run_atmosphere_test_matrix.py-6066-            from tests.test_cases.dcmip2025 import dcmip25_tc3_init
./scripts/matrix/run_atmosphere_test_matrix.py-6067-            state, hcoord, tmetric, small_grid = dcmip25_tc3_init(
./scripts/matrix/run_atmosphere_test_matrix.py-6068-                grid, n_levels=nlev)
./scripts/matrix/run_atmosphere_test_matrix.py-6069-            grid = small_grid
./scripts/matrix/run_atmosphere_test_matrix.py-6070-            #
./scripts/matrix/run_atmosphere_test_matrix.py:6071:            # ⚠️ new_test_dycores iter-108 CAUTION (CONFIRMED by iter-123):
./scripts/matrix/run_atmosphere_test_matrix.py-6072-            # TC3 cube BLOWS UP at full mode at step 2250 (day 0.01,
./scripts/matrix/run_atmosphere_test_matrix.py-6073-            # 8.3 min sim time) with `metric 1271.0 > threshold 1000.0`
./scripts/matrix/run_atmosphere_test_matrix.py-6074-            # (max|w| exceeds threshold).  Same iter-12..17 NH bundle
./scripts/matrix/run_atmosphere_test_matrix.py-6075-            # that gives PASS at quick mode (|w|=7.36 m/s per iter-7
./scripts/matrix/run_atmosphere_test_matrix.py-6076-            # claim) is INSUFFICIENT for full-duration cube stability.
--
./scripts/matrix/run_atmosphere_test_matrix.py-8069-                                                 C72, dt=100 at C96;
./scripts/matrix/run_atmosphere_test_matrix.py-8070-                                                 NaN day 22.5 at C96
./scripts/matrix/run_atmosphere_test_matrix.py-8071-                                                 30d per iter 79)
./scripts/matrix/run_atmosphere_test_matrix.py-8072-                                  very_long_time iter-80 (dt=67 at
./scripts/matrix/run_atmosphere_test_matrix.py-8073-                                                 C72, dt=50 at C96;
./scripts/matrix/run_atmosphere_test_matrix.py:8074:                                                 iter-99 CONFIRMED
./scripts/matrix/run_atmosphere_test_matrix.py-8075-                                                 30d finite at C96
./scripts/matrix/run_atmosphere_test_matrix.py-8076-                                                 max|u|=20.14)
./scripts/matrix/run_atmosphere_test_matrix.py-8077-                                  auto           RECOMMENDED: short_time
./scripts/matrix/run_atmosphere_test_matrix.py-8078-                                                 at n<96, very_long_time
./scripts/matrix/run_atmosphere_test_matrix.py-8079-                                                 at n>=96.  Validated
--
./scripts/run/run_rce_mpi_long.py-517-    )
./scripts/run/run_rce_mpi_long.py-518-
./scripts/run/run_rce_mpi_long.py-519-    # iter-181 theta' noise seed in the lowest 4 levels (RCEMIP /
./scripts/run/run_rce_mpi_long.py-520-    # Wing 2018 standard symmetry-breaker). iter-212 update: the
./scripts/run/run_rce_mpi_long.py-521-    # original iter-181 claim that theta' was safer than qv noise
./scripts/run/run_rce_mpi_long.py:522:    # because it doesn't enter the LW optical depth was REFUTED at
./scripts/run/run_rce_mpi_long.py-523-    # the iter-212 smoke — theta' nonzero amplitudes blow up via a
./scripts/run/run_rce_mpi_long.py-524-    # DIFFERENT mechanism (direct buoyancy injection, not radiation
./scripts/run/run_rce_mpi_long.py-525-    # feedback). Same F11 dx=4 km wall, just from a different
./scripts/run/run_rce_mpi_long.py-526-    # physics pathway.
./scripts/run/run_rce_mpi_long.py-527-    # iter-203: added --theta-noise-mode to pick the perturbation
--
scripts/validate/land_carbon_equilibrium.py-218-       litterfall, temperature-modified turnover) become stationary.
scripts/validate/land_carbon_equilibrium.py-219-    2. Solve the LINEAR slow-pool steady state analytically from those mean
scripts/validate/land_carbon_equilibrium.py-220-       fluxes — for a first-order pool ``dC/dt = I - k·C``, the turnover ``k``
scripts/validate/land_carbon_equilibrium.py-221-       is C-independent, so ``C_eq = I / k = C_spinup · (I_annual / loss_annual)``
scripts/validate/land_carbon_equilibrium.py-222-       — and reset wood + SOM to it.
scripts/validate/land_carbon_equilibrium.py:223:    3. Run ``n_verify`` more years to CONFIRM the analytic equilibrium holds
scripts/validate/land_carbon_equilibrium.py-224-       (the reported drift → 0); the returned diagnostics are this verified
scripts/validate/land_carbon_equilibrium.py-225-       equilibrium segment.
scripts/validate/land_carbon_equilibrium.py-226-
scripts/validate/land_carbon_equilibrium.py-227-    Returns (annual_diag over the verify segment, final_state, final_carbon).
scripts/validate/land_carbon_equilibrium.py-228-    """
--
scripts/validate/compare_amip_era5.py-297-    ``expected_vertical_coord`` (the operator's ``--vertical-coord``): when the
scripts/validate/compare_amip_era5.py-298-    restart RECORDS the run's coordinate (``loaded_config.grid.vertical_coord``),
scripts/validate/compare_amip_era5.py-299-    a mismatch is FAILED LOUDLY — placing the state on the wrong pressure levels
scripts/validate/compare_amip_era5.py-300-    (the warning the flag's help only described) would silently bias the compare.
scripts/validate/compare_amip_era5.py-301-    Defensive: skipped when the recorded coordinate is unavailable (legacy / no
scripts/validate/compare_amip_era5.py:302:    config), so it only raises on a CONFIRMED mismatch.
scripts/validate/compare_amip_era5.py-303-    """
scripts/validate/compare_amip_era5.py-304-    from legoesm.driver.restart import load_restart
scripts/validate/compare_amip_era5.py-305-    from legoesm.training.compare_reanalysis import grid_winds_from_spectral
scripts/validate/compare_amip_era5.py-306-
scripts/validate/compare_amip_era5.py-307-    loaded = load_restart(restart_path, grid, sigma, strict=True)
--
scripts/validate/validate_tiled_fv3_sw_multinode.py-3-
scripts/validate/validate_tiled_fv3_sw_multinode.py-4-Runs ``make_tiled_fv3_sw_tendencies_stage_2d`` under REAL multi-controller
scripts/validate/validate_tiled_fv3_sw_multinode.py-5-``jax.distributed`` across N nodes — so the in-stage halo ``lax.ppermute``
scripts/validate/validate_tiled_fv3_sw_multinode.py-6-exchanges (B scalar, the shared cc-wind vector, the du/dv tendency vector) and
scripts/validate/validate_tiled_fv3_sw_multinode.py-7-the per-tile reassembly become genuine CROSS-PROCESS (and cross-NODE)
scripts/validate/validate_tiled_fv3_sw_multinode.py:8:collective-permute, not the single-process ``XLA_FLAGS=--xla_force_host_platform_
scripts/validate/validate_tiled_fv3_sw_multinode.py-9-device_count=N`` virtual-device path the pytest gate uses.  Each process then
scripts/validate/validate_tiled_fv3_sw_multinode.py-10-self-validates: its LOCAL tile output must be bit-identical (rel < 1e-9) to the
scripts/validate/validate_tiled_fv3_sw_multinode.py-11-serial global ``fv3_sw_tendencies`` computed on the (replicated) inputs.  No
scripts/validate/validate_tiled_fv3_sw_multinode.py-12-cross-process gather is needed — every process owns one (face, tile_i, tile_j)
scripts/validate/validate_tiled_fv3_sw_multinode.py-13-tile and checks it against the matching global slice; ALL processes passing is
--
scripts/validate/validate_tiled_fv3_sw_multinode.py-47-    gdev, ldev = jax.devices(), jax.local_devices()
scripts/validate/validate_tiled_fv3_sw_multinode.py-48-    if pidx == 0:
scripts/validate/validate_tiled_fv3_sw_multinode.py-49-        log(f"process {pidx}/{pcount}  global_devices={len(gdev)} "
scripts/validate/validate_tiled_fv3_sw_multinode.py-50-            f"local_devices={len(ldev)}")
scripts/validate/validate_tiled_fv3_sw_multinode.py-51-    # This validator's POINT is real multi-process execution (the in-stage
scripts/validate/validate_tiled_fv3_sw_multinode.py:52:    # ppermutes -> cross-process collective-permute).  A single-process run with
scripts/validate/validate_tiled_fv3_sw_multinode.py-53-    # N virtual devices would pass vacuously, so require >1 process unless an
scripts/validate/validate_tiled_fv3_sw_multinode.py-54-    # explicit escape is set (codex audit LOW).
scripts/validate/validate_tiled_fv3_sw_multinode.py-55-    if pcount <= 1 and os.environ.get("ALLOW_SINGLE_PROCESS") != "1":
scripts/validate/validate_tiled_fv3_sw_multinode.py-56-        log("REFUSING single-process run: this validates MULTI-process/node "
scripts/validate/validate_tiled_fv3_sw_multinode.py-57-            "execution.  Launch via srun -n <6*kt*kt>, or set "
--
scripts/validate/validate_tiled_fv3_sw_multinode.py-128-            import re
scripts/validate/validate_tiled_fv3_sw_multinode.py-129-            hlo = jax.jit(
scripts/validate/validate_tiled_fv3_sw_multinode.py-130-                lambda a, b, c, d: stage(a, b, c, d)).lower(
scripts/validate/validate_tiled_fv3_sw_multinode.py-131-                to_global(h), to_global(u_d), to_global(v_d),
scripts/validate/validate_tiled_fv3_sw_multinode.py-132-                to_global(h_s)).compile().as_text()
scripts/validate/validate_tiled_fv3_sw_multinode.py:133:            cp = len(re.findall(r"\bcollective-permute\b", hlo))
scripts/validate/validate_tiled_fv3_sw_multinode.py:134:            log(f"HLO: collective-permute={cp} (cross-process halo exchange)")
scripts/validate/validate_tiled_fv3_sw_multinode.py-135-        except Exception as exc:  # noqa: BLE001
scripts/validate/validate_tiled_fv3_sw_multinode.py-136-            log(f"HLO inspect skipped: {type(exc).__name__}: {exc}")
scripts/validate/validate_tiled_fv3_sw_multinode.py-137-
scripts/validate/validate_tiled_fv3_sw_multinode.py-138-    # Each process self-validates its LOCAL tile(s) vs the serial global slice.
scripts/validate/validate_tiled_fv3_sw_multinode.py-139-    def _worst(t_arr, g_full, stag_i, stag_j):
--
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-337-    vm = cfg.physics.vertical_mixing
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-338-    if (tke_mxl_choice is not None
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-339-            and tke_mxl_choice != vm.tke.tke_mxl_choice):
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-340-        # Generic experimentation override. The recipe ships the faithful
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-341-        # tke_mxl_choice=1 (Veros global_1deg.py:64) which now runs natively
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py:342:        # — the distance-to-boundary cap (veros_mxl_choice1_boundary_cap)
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-343-        # makes it debt-safe, so NO fallback is needed. This knob only exists
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-344-        # to A/B the choice=2 recursion against choice=1; any use is a LOUD
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-345-        # deviation from the oracle, recorded in the output config.
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-346-        print(f"!! DEVIATION (experimentation): tke_mxl_choice={tke_mxl_choice!r} "
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-347-              f"(Veros-faithful recipe: {vm.tke.tke_mxl_choice!r})")
--
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-497-                    choices=("none", "superbee"))
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-498-    ap.add_argument("--tke-mxl-choice", type=int, default=None,
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-499-                    choices=(1, 2),
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-500-                    help="experimentation override for TKEConfig.tke_mxl_choice; "
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-501-                         "the recipe ships the faithful 1 (now debt-safe via "
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py:502:                         "the distance-to-boundary cap), so this is NOT needed "
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-503-                         "for stability — only to A/B against choice=2")
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-504-    args = ap.parse_args()
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-505-
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-506-    import jax
scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-507-    jax.config.update("jax_enable_x64", True)
--
scripts/validate/ocean_fidelity/run_acc_freerun.py-232-     weak). BUT the faithful drag OVER-DAMPS the transport: @1yr it gives -83% (18 vs
scripts/validate/ocean_fidelity/run_acc_freerun.py-233-     110 Sv) vs the mis-mapped r=1e-5's +58% -> legoESM's barotropic transport
scripts/validate/ocean_fidelity/run_acc_freerun.py-234-     over-responds to bottom drag vs Veros. The genuine deeper difference is the
scripts/validate/ocean_fidelity/run_acc_freerun.py-235-     BAROTROPIC FORMULATION (free surface vs Veros rigid-lid/streamfunction); the old
scripts/validate/ocean_fidelity/run_acc_freerun.py-236-     r=1e-5 masked it via compensating errors. (30d looked like a fix: KE +233%->+27%;
scripts/validate/ocean_fidelity/run_acc_freerun.py:237:     1yr revealed the over-damping.) CONFIRMED: a rigid-lid option (--barotropic-solver
scripts/validate/ocean_fidelity/run_acc_freerun.py-238-     rigid_lid) is now built and RESTORES an O(100 Sv) ACC vs the free surface's collapsed
scripts/validate/ocean_fidelity/run_acc_freerun.py-239-     18 Sv at the SAME faithful drag -> the barotropic formulation was the dominant control.
scripts/validate/ocean_fidelity/run_acc_freerun.py-240-  1. Time integrator: NOW MATCHED (#44). --outer-integrator ab2 is the faithful Veros
scripts/validate/ocean_fidelity/run_acc_freerun.py-241-     scheme — AB2 on the EXPLICIT tendency + implicit vertical mixing applied ONCE
scripts/validate/ocean_fidelity/run_acc_freerun.py-242-     (tracers temp[taup1]=temp[tau]+dt_tracer*((1.5+eps)*dtemp[tau]-(0.5+eps)*dtemp[taum1]),
--
./packages/coupler/legoesm/driver/model_driver.py-9345-        seg_len = max(1, int(86400.0 / DT))
./packages/coupler/legoesm/driver/model_driver.py-9346-        current_step = start_step
./packages/coupler/legoesm/driver/model_driver.py-9347-        status = "COMPLETED"
./packages/coupler/legoesm/driver/model_driver.py-9348-        seg_idx = -1
./packages/coupler/legoesm/driver/model_driver.py-9349-        # #921: prime every NCCL clique this step uses (the halo
./packages/coupler/legoesm/driver/model_driver.py:9350:        # collective-permutes + the target-mass / moisture-fixer psums) in a
./packages/coupler/legoesm/driver/model_driver.py-9351-        # fixed, rank-independent order before the first real step, so
./packages/coupler/legoesm/driver/model_driver.py-9352-        # multi-process (route-B) comm-init cannot deadlock.  No-op
./packages/coupler/legoesm/driver/model_driver.py-9353-        # single-process (CPU-virtual / single-GPU) — those lanes are unchanged.
./packages/coupler/legoesm/driver/model_driver.py-9354-        from legoesm.parallel.tiled_production_cdgrid import (
./packages/coupler/legoesm/driver/model_driver.py-9355-            warmup_tiled_cube_comms,
--
./packages/coupler/legoesm/driver/model_driver.py-9607-            })
./packages/coupler/legoesm/driver/model_driver.py-9608-        template = self.state
./packages/coupler/legoesm/driver/model_driver.py-9609-        blocked = enter(self.state)
./packages/coupler/legoesm/driver/model_driver.py-9610-
./packages/coupler/legoesm/driver/model_driver.py-9611-        # #921: prime every NCCL clique the blocked step uses (the halo
./packages/coupler/legoesm/driver/model_driver.py:9612:        # collective-permutes + the in-stage mass-fixer psum) in a fixed,
./packages/coupler/legoesm/driver/model_driver.py-9613-        # rank-independent order before the first real step, so multi-process
./packages/coupler/legoesm/driver/model_driver.py-9614-        # (route-B one-process-per-GPU) comm-init cannot deadlock.  No-op
./packages/coupler/legoesm/driver/model_driver.py-9615-        # single-process (CPU-virtual / single-GPU) — those lanes are unchanged.
./packages/coupler/legoesm/driver/model_driver.py-9616-        from legoesm.parallel.tiled_production_cdgrid import (
./packages/coupler/legoesm/driver/model_driver.py-9617-            warmup_tiled_cube_comms,
--
./scripts/validate/validate_tiled_fv3_sw_multinode.py-3-
./scripts/validate/validate_tiled_fv3_sw_multinode.py-4-Runs ``make_tiled_fv3_sw_tendencies_stage_2d`` under REAL multi-controller
./scripts/validate/validate_tiled_fv3_sw_multinode.py-5-``jax.distributed`` across N nodes — so the in-stage halo ``lax.ppermute``
./scripts/validate/validate_tiled_fv3_sw_multinode.py-6-exchanges (B scalar, the shared cc-wind vector, the du/dv tendency vector) and
./scripts/validate/validate_tiled_fv3_sw_multinode.py-7-the per-tile reassembly become genuine CROSS-PROCESS (and cross-NODE)
./scripts/validate/validate_tiled_fv3_sw_multinode.py:8:collective-permute, not the single-process ``XLA_FLAGS=--xla_force_host_platform_
./scripts/validate/validate_tiled_fv3_sw_multinode.py-9-device_count=N`` virtual-device path the pytest gate uses.  Each process then
./scripts/validate/validate_tiled_fv3_sw_multinode.py-10-self-validates: its LOCAL tile output must be bit-identical (rel < 1e-9) to the
./scripts/validate/validate_tiled_fv3_sw_multinode.py-11-serial global ``fv3_sw_tendencies`` computed on the (replicated) inputs.  No
./scripts/validate/validate_tiled_fv3_sw_multinode.py-12-cross-process gather is needed — every process owns one (face, tile_i, tile_j)
./scripts/validate/validate_tiled_fv3_sw_multinode.py-13-tile and checks it against the matching global slice; ALL processes passing is
--
./scripts/validate/validate_tiled_fv3_sw_multinode.py-47-    gdev, ldev = jax.devices(), jax.local_devices()
./scripts/validate/validate_tiled_fv3_sw_multinode.py-48-    if pidx == 0:
./scripts/validate/validate_tiled_fv3_sw_multinode.py-49-        log(f"process {pidx}/{pcount}  global_devices={len(gdev)} "
./scripts/validate/validate_tiled_fv3_sw_multinode.py-50-            f"local_devices={len(ldev)}")
./scripts/validate/validate_tiled_fv3_sw_multinode.py-51-    # This validator's POINT is real multi-process execution (the in-stage
./scripts/validate/validate_tiled_fv3_sw_multinode.py:52:    # ppermutes -> cross-process collective-permute).  A single-process run with
./scripts/validate/validate_tiled_fv3_sw_multinode.py-53-    # N virtual devices would pass vacuously, so require >1 process unless an
./scripts/validate/validate_tiled_fv3_sw_multinode.py-54-    # explicit escape is set (codex audit LOW).
./scripts/validate/validate_tiled_fv3_sw_multinode.py-55-    if pcount <= 1 and os.environ.get("ALLOW_SINGLE_PROCESS") != "1":
./scripts/validate/validate_tiled_fv3_sw_multinode.py-56-        log("REFUSING single-process run: this validates MULTI-process/node "
./scripts/validate/validate_tiled_fv3_sw_multinode.py-57-            "execution.  Launch via srun -n <6*kt*kt>, or set "
--
./scripts/validate/validate_tiled_fv3_sw_multinode.py-128-            import re
./scripts/validate/validate_tiled_fv3_sw_multinode.py-129-            hlo = jax.jit(
./scripts/validate/validate_tiled_fv3_sw_multinode.py-130-                lambda a, b, c, d: stage(a, b, c, d)).lower(
./scripts/validate/validate_tiled_fv3_sw_multinode.py-131-                to_global(h), to_global(u_d), to_global(v_d),
./scripts/validate/validate_tiled_fv3_sw_multinode.py-132-                to_global(h_s)).compile().as_text()
./scripts/validate/validate_tiled_fv3_sw_multinode.py:133:            cp = len(re.findall(r"\bcollective-permute\b", hlo))
./scripts/validate/validate_tiled_fv3_sw_multinode.py:134:            log(f"HLO: collective-permute={cp} (cross-process halo exchange)")
./scripts/validate/validate_tiled_fv3_sw_multinode.py-135-        except Exception as exc:  # noqa: BLE001
./scripts/validate/validate_tiled_fv3_sw_multinode.py-136-            log(f"HLO inspect skipped: {type(exc).__name__}: {exc}")
./scripts/validate/validate_tiled_fv3_sw_multinode.py-137-
./scripts/validate/validate_tiled_fv3_sw_multinode.py-138-    # Each process self-validates its LOCAL tile(s) vs the serial global slice.
./scripts/validate/validate_tiled_fv3_sw_multinode.py-139-    def _worst(t_arr, g_full, stag_i, stag_j):
--
./scripts/validate/land_carbon_equilibrium.py-218-       litterfall, temperature-modified turnover) become stationary.
./scripts/validate/land_carbon_equilibrium.py-219-    2. Solve the LINEAR slow-pool steady state analytically from those mean
./scripts/validate/land_carbon_equilibrium.py-220-       fluxes — for a first-order pool ``dC/dt = I - k·C``, the turnover ``k``
./scripts/validate/land_carbon_equilibrium.py-221-       is C-independent, so ``C_eq = I / k = C_spinup · (I_annual / loss_annual)``
./scripts/validate/land_carbon_equilibrium.py-222-       — and reset wood + SOM to it.
./scripts/validate/land_carbon_equilibrium.py:223:    3. Run ``n_verify`` more years to CONFIRM the analytic equilibrium holds
./scripts/validate/land_carbon_equilibrium.py-224-       (the reported drift → 0); the returned diagnostics are this verified
./scripts/validate/land_carbon_equilibrium.py-225-       equilibrium segment.
./scripts/validate/land_carbon_equilibrium.py-226-
./scripts/validate/land_carbon_equilibrium.py-227-    Returns (annual_diag over the verify segment, final_state, final_carbon).
./scripts/validate/land_carbon_equilibrium.py-228-    """
--
./scripts/validate/compare_amip_era5.py-297-    ``expected_vertical_coord`` (the operator's ``--vertical-coord``): when the
./scripts/validate/compare_amip_era5.py-298-    restart RECORDS the run's coordinate (``loaded_config.grid.vertical_coord``),
./scripts/validate/compare_amip_era5.py-299-    a mismatch is FAILED LOUDLY — placing the state on the wrong pressure levels
./scripts/validate/compare_amip_era5.py-300-    (the warning the flag's help only described) would silently bias the compare.
./scripts/validate/compare_amip_era5.py-301-    Defensive: skipped when the recorded coordinate is unavailable (legacy / no
./scripts/validate/compare_amip_era5.py:302:    config), so it only raises on a CONFIRMED mismatch.
./scripts/validate/compare_amip_era5.py-303-    """
./scripts/validate/compare_amip_era5.py-304-    from legoesm.driver.restart import load_restart
./scripts/validate/compare_amip_era5.py-305-    from legoesm.training.compare_reanalysis import grid_winds_from_spectral
./scripts/validate/compare_amip_era5.py-306-
./scripts/validate/compare_amip_era5.py-307-    loaded = load_restart(restart_path, grid, sigma, strict=True)
--
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-337-    vm = cfg.physics.vertical_mixing
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-338-    if (tke_mxl_choice is not None
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-339-            and tke_mxl_choice != vm.tke.tke_mxl_choice):
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-340-        # Generic experimentation override. The recipe ships the faithful
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-341-        # tke_mxl_choice=1 (Veros global_1deg.py:64) which now runs natively
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py:342:        # — the distance-to-boundary cap (veros_mxl_choice1_boundary_cap)
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-343-        # makes it debt-safe, so NO fallback is needed. This knob only exists
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-344-        # to A/B the choice=2 recursion against choice=1; any use is a LOUD
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-345-        # deviation from the oracle, recorded in the output config.
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-346-        print(f"!! DEVIATION (experimentation): tke_mxl_choice={tke_mxl_choice!r} "
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-347-              f"(Veros-faithful recipe: {vm.tke.tke_mxl_choice!r})")
--
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-497-                    choices=("none", "superbee"))
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-498-    ap.add_argument("--tke-mxl-choice", type=int, default=None,
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-499-                    choices=(1, 2),
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-500-                    help="experimentation override for TKEConfig.tke_mxl_choice; "
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-501-                         "the recipe ships the faithful 1 (now debt-safe via "
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py:502:                         "the distance-to-boundary cap), so this is NOT needed "
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-503-                         "for stability — only to A/B against choice=2")
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-504-    args = ap.parse_args()
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-505-
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-506-    import jax
./scripts/validate/ocean_fidelity/run_global_1deg_freerun.py-507-    jax.config.update("jax_enable_x64", True)
--
scripts/plot/plot_campaign_bias_trajectory.py-1-"""Plot the LES-informed correction campaign's bias trajectory from its output JSON.
scripts/plot/plot_campaign_bias_trajectory.py-2-
scripts/plot/plot_campaign_bias_trajectory.py-3-After the (multi-day) campaign writes its corrected-config JSON
scripts/plot/plot_campaign_bias_trajectory.py-4-(``run_correction_campaign.py --out <file>``), this renders the done-criterion's
scripts/plot/plot_campaign_bias_trajectory.py:5:VISUAL CONFIRMATION — did the LES-informed coefficient updates LOWER the bias? — as
scripts/plot/plot_campaign_bias_trajectory.py-6-a two-panel PNG:
scripts/plot/plot_campaign_bias_trajectory.py-7-
scripts/plot/plot_campaign_bias_trajectory.py-8-  1. the global combined bias per round (baseline → updated, with ACCEPTED rounds
scripts/plot/plot_campaign_bias_trajectory.py-9-     marked; the monotonic gate keeps only bias-lowering rounds), plus the
scripts/plot/plot_campaign_bias_trajectory.py-10-     campaign-START → final markers;
--
./scripts/validate/ocean_fidelity/run_acc_freerun.py-232-     weak). BUT the faithful drag OVER-DAMPS the transport: @1yr it gives -83% (18 vs
./scripts/validate/ocean_fidelity/run_acc_freerun.py-233-     110 Sv) vs the mis-mapped r=1e-5's +58% -> legoESM's barotropic transport
./scripts/validate/ocean_fidelity/run_acc_freerun.py-234-     over-responds to bottom drag vs Veros. The genuine deeper difference is the
./scripts/validate/ocean_fidelity/run_acc_freerun.py-235-     BAROTROPIC FORMULATION (free surface vs Veros rigid-lid/streamfunction); the old
./scripts/validate/ocean_fidelity/run_acc_freerun.py-236-     r=1e-5 masked it via compensating errors. (30d looked like a fix: KE +233%->+27%;
./scripts/validate/ocean_fidelity/run_acc_freerun.py:237:     1yr revealed the over-damping.) CONFIRMED: a rigid-lid option (--barotropic-solver
./scripts/validate/ocean_fidelity/run_acc_freerun.py-238-     rigid_lid) is now built and RESTORES an O(100 Sv) ACC vs the free surface's collapsed
./scripts/validate/ocean_fidelity/run_acc_freerun.py-239-     18 Sv at the SAME faithful drag -> the barotropic formulation was the dominant control.
./scripts/validate/ocean_fidelity/run_acc_freerun.py-240-  1. Time integrator: NOW MATCHED (#44). --outer-integrator ab2 is the faithful Veros
./scripts/validate/ocean_fidelity/run_acc_freerun.py-241-     scheme — AB2 on the EXPLICIT tendency + implicit vertical mixing applied ONCE
./scripts/validate/ocean_fidelity/run_acc_freerun.py-242-     (tracers temp[taup1]=temp[tau]+dt_tracer*((1.5+eps)*dtemp[tau]-(0.5+eps)*dtemp[taum1]),
--
./CLAUDE.md-163-- Dycore: Williamson, Galewsky, Jablonowski-Williamson, DCMIP, Held-Suarez, ocean benchmarks.
./CLAUDE.md-164-- Conservation/reductions/coupler: mass+energy diagnostics.
./CLAUDE.md-165-- Parallel: unsharded vs sharded, single-rank vs MPI.
./CLAUDE.md-166-- Too expensive: say what ran/didn't, residual risk.
./CLAUDE.md-167-- **CRITICAL — Controlled comparison: change ONE variable, hold the eval protocol FIXED to the baseline.** To claim a change (resolution, params, scheme, days) improved/degraded/"is comparable" vs a prior result, keep EVERYTHING else byte-identical to that baseline: forcing data + its sampling (years/days/hours/climatology), evaluation grid, metric definition (bias vs RMSE), region masks, timestepping. A metric that moved because the protocol/sampling changed is a **CONFOUND, not a result** — NEVER compare a number computed on one sampling/grid/metric to a number from another and call the difference an effect. If a resource limit (network cap, compute, time) forces a lighter or different sampling, **re-run the BASELINE at that SAME sampling before comparing** — a fresh baseline is cheap insurance; a confounded claim is not. Before writing "improved"/"degraded"/"better"/"comparable" vs any earlier number, explicitly confirm the two configs differ ONLY in the variable under test; if you cannot, say so and claim NO direction. Report the full config (data+sampling, grid, days/steps, params, metric) next to every number so the reader knows exactly what is being compared. Assuming two runs are comparable when the setup drifted is the error that turns "we improved it" into "we degraded it."
./CLAUDE.md:168:- **CRITICAL — Precision gate for comparison/skill/causal claims (do NOT be sloppy — 2026-07-20 EC-site lesson).** (a) **Harness self-check FIRST**: before reporting a NEW scheme's skill vs a validated baseline, run the KNOWN baseline through your OWN harness and confirm it reproduces the baseline's established/published number (±small tol). If your harness scores a validated scheme wildly off — e.g. two-leaf H looked "broken" (NSE≈−1) when the paper figure tracks obs — the HARNESS is wrong; fix it BEFORE any new-scheme claim. (b) **Match the reference metric EXACTLY**: metric aggregation (daily-mean vs half-hourly point-wise NSE), window (multi-year JJA vs one summer), masks (valid-forcing) are ALL part of the protocol. Report the SAME metric the reference used, plus any alternative, WITH the sample N. A single-window point-wise metric is NOT a skill verdict, and never rank schemes by a metric that punishes one scheme's known artifact (point-wise scatter/spikes) while ignoring the dimension of interest (mean diurnal shape). (c) **Never let a QC mask flatter one side**: if you drop a scheme's unphysical spikes from scoring, report BOTH the failures-penalized primary score AND the separately-labeled plausible-only sensitivity score. (d) **Do NOT INFER causal origin — INSTRUMENT it**: "the spikes come from X" requires LOGGING X and the alternatives (raw vs intermediate vs final), not deduction from reading one code path; "correctly hooked up" requires reading the FULL path, not the entry point. (e) **Label every claim CONFIRMED (evidence shown) vs PLAUSIBLE (inferred)** — adversarial review WILL refute over-confident inferences; state uncertainty up front rather than presenting a verdict table that a five-minute check overturns.
./CLAUDE.md-169-- **CRITICAL — Visual verify spatial/grid artifacts**: passing tests+norms NECESSARY ≠ SUFFICIENT for cubed-sphere ops, halo exchange, diffusion coeffs, grid metrics. Edge artifacts/cube imprint/grid-scale noise only detected visually (v-wind W2, wind_speed W5). Run `--only sw --grid cubed_sphere --quick` + inspect PNGs vs baseline. Norms can improve while artifacts worsen. Never claim "tests pass, edge fixed" from pytest alone.
./CLAUDE.md-170-- **Diffusion sensitivity**: div damping + hyperdiff AMPLIFY halo errors at cubed-sphere face boundaries. Check W2 v-wind visually when touching `_hyperdiff_cube`, `_div_damp_cube`, diffusion params.
./CLAUDE.md-171-- **Visual-regression gate (cube imprint)**: `scripts/validate/visual_regression.py --check` numericises the W2 v-wind cube-imprint check (SSIM + per-panel perceptual hash + edge-artifact ratio vs tiny committed ref in `tests/visual_baselines/`). Deterministic metric math gated in CI (`tests/test_visual_regression_metrics.py`); full cube-SW `--check` runs as a NIGHTLY non-blocking CI job until tolerances are calibrated across CI hardware. Tiny numeric baselines (.npy+json) ARE tracked — the one carve-out to "no tracked visual baselines".
./CLAUDE.md-172-- **CRITICAL — VALIDATE THE INSTRUMENT BEFORE QUOTING ITS NUMBER (2026-07-25 duogrid lesson: 8 confident claims, all retracted).** A diagnostic script is UNTRUSTED CODE until it passes its own controls. Never state a finding — never write "measured", "confirmed", "proven", "VERDICT" — from a probe's first output. **Before quoting any diagnostic number, run these five checks and say in the message that you ran them:**
./CLAUDE.md-173-  1. **Right conserved/invariant quantity?** Budget what the SYSTEM conserves, not a convenient proxy. (Failed: reported "vertex creates energy" from **KE alone** — KE is NOT conserved in shallow water, it trades with PE. Total `E=∫area(½h|V|²+½gh²)` reversed the sign of the conclusion.)
./CLAUDE.md-174-  2. **Same transform / units / staggering on BOTH sides?** Two "A-grid winds" from different operators are DIFFERENT QUANTITIES. (Failed: ours `c2l_ord2` vs oracle `C2L_ORD=4` — the SAME raw state gave 1.96e-2 vs 5.79e-2, a 3× swing that WAS the reported effect. Also: never budget across a stage boundary where the state changes representation — mid-step FV3 winds are in circulation form, which produced ±5.6e10 garbage.)
./CLAUDE.md-175-  3. **Same time, resolution, config?** Index by MATCHED TIME, not frame number. (Failed: mapped day→frame as `round(day)-1` against an HOURLY file, comparing our day-1 to their hour-1; and quoted a **C12** wedge gain (~300×) as the mechanism for a **C48** instability, where it is ~124×.)
./CLAUDE.md-176-  4. **Is the metric measuring what its name says?** Prove it on a synthetic case with a KNOWN answer before use. (Failed: called `mean|f−4-neighbour-mean|` a "2Δx grid-scale" measure — it is a high-pass/curvature residual that a merely sharper SMOOTH feature reproduces. Failed: a "gain" probe that re-filled a FIXED source, which is trivially 1.0000 by construction.)
./CLAUDE.md-177-  5. **Can the reduction support the claim?** `max` over tiles/corners/components taken independently per run can peak at DIFFERENT physical locations; a max-of-per-tile-means is not a global mean. Keep argmax metadata and map to a common physical location before claiming "localized".
./CLAUDE.md-178-  Plus: **diff ARRAYS, never printed summaries** (claimed "bit-identical ⇒ deterministic, not chaos"; the arrays actually differed by 9e-6 — only the rounded printout matched). **Never let a probe print its own verdict** ("=> the growth is REAL") — the interpretation belongs in the analysis after the controls pass, not baked into the tool where it gets echoed back as evidence. **`nanmax`/`nanmean`/`nansum` hide failures** — make NaN and missing frames FATAL. **Record every effective flag, env var and git SHA in each artifact**; a default `--n 36` silently mis-slicing a C48 file runs fine and lies.
./CLAUDE.md:179:- **CRITICAL — LABEL EVERY CLAIM, AND PREFER RETRACTING EARLY.** Tag each statement **CONFIRMED** (control-passed evidence shown, instrument validated) vs **PLAUSIBLE** (inferred / single-run / uncontrolled). A chain of PLAUSIBLE steps is not CONFIRMED. When a later measurement contradicts an earlier claim, **retract it loudly and immediately in the same message and in the memory file** — do not quietly move on, because stale confident claims get built on. Run the codex adversarial review on the DIAGNOSTIC TOOLING, not just the model code: the instruments decide what you believe, so a bug there manufactures a confident wrong physics conclusion. Cheap self-check before any big claim: *"what measurement would make this false, and did I run it?"* If the answer is no, the claim is PLAUSIBLE at best.
./CLAUDE.md-180-
./CLAUDE.md-181-## Domain Architect vs Syntax Engine (AI guardrails)
./CLAUDE.md-182-See `docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md`. Doctrine: the human dictates the *logic* (units, signs, conserved qty, valid scheme sets, references, acceptance criteria); AI fills the *body*; every declared invariant is checked **mechanically** so violations fail LOUDLY. Each gate is a **tripwire, not a proof** and ships a synthetic-violation self-test (provably non-vacuous). NON-NEGOTIABLE harness (extend, never weaken; budgets/TODOs shrink only):
./CLAUDE.md-183-- **Spec-first physics contracts**: every physics scheme module declares `__physics_contract__` (units/signs/conserves/differentiable/reference/idealized_test); `tests/test_physics_contracts.py` partitions all `*/physics/*.py` into EXCLUDED / CONTRACT_TODO(shrink-only) / annotated — a NEW physics file must ship a contract or be classified. Author the contract + acceptance test BEFORE the body.
./CLAUDE.md-184-- **Atmospheric parameterization edits are high-risk**: run codex adversarial review, #477 truth-tier tendency validators, conservation tests, and the equilibrium-SCM-RCE realism harness before merge. A NEW scheme must pass the RCE realism gate or enter its shrink-only TODO partition with a reason.
--
./scripts/plot/plot_campaign_bias_trajectory.py-1-"""Plot the LES-informed correction campaign's bias trajectory from its output JSON.
./scripts/plot/plot_campaign_bias_trajectory.py-2-
./scripts/plot/plot_campaign_bias_trajectory.py-3-After the (multi-day) campaign writes its corrected-config JSON
./scripts/plot/plot_campaign_bias_trajectory.py-4-(``run_correction_campaign.py --out <file>``), this renders the done-criterion's
./scripts/plot/plot_campaign_bias_trajectory.py:5:VISUAL CONFIRMATION — did the LES-informed coefficient updates LOWER the bias? — as
./scripts/plot/plot_campaign_bias_trajectory.py-6-a two-panel PNG:
./scripts/plot/plot_campaign_bias_trajectory.py-7-
./scripts/plot/plot_campaign_bias_trajectory.py-8-  1. the global combined bias per round (baseline → updated, with ACCEPTED rounds
./scripts/plot/plot_campaign_bias_trajectory.py-9-     marked; the monotonic gate keeps only bias-lowering rounds), plus the
./scripts/plot/plot_campaign_bias_trajectory.py-10-     campaign-START → final markers;
--
./packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-645-
./packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-646-    ADV-SPLIT #86 (codex iter-68): a 2nd-order CENTERED, NON-diffusive momentum
./packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-647-    scheme matching SAM's face reconstruction (``advect2_mom_xy.f90:27``:
./packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-648-    ``flux=0.25·(c_face)·(φ_face)`` — no upwind bias, no flux limiter). The TVD
./packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-649-    van_leer default adds limiter diffusion that SUPPRESSES convective updraft
./packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py:650:    cores / w-variance tails (CONFIRMED: identical-IC max|w| caps ~2.6 m/s vs
./packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-651-    low-diffusion ~5-15; centered restores ~5 in a gray burst). For SAM-faithful
./packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-652-    convective EXTREMES the momentum legs (u/v/w) use this scheme; the SCALAR legs
./packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-653-    keep van_leer (monotone ≈ MPDATA, positivity). NOTE (codex iter-68 [S1]): this
./packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-654-    uses the ADVECTIVE form ``−d(uf)/dx + f·du/dx`` (same identity as van_leer/
./packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-655-    weno5 — preserves a constant f under divergent flow), which is NOT proven
--
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-439-    STEP, and the no-full-face-all-gather property holds for the WHOLE
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-440-    compiled program (gated on HLO text by
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-441-    ``tests/parallel/test_cube_tile_native_segment.py`` — zero
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-442-    ``all-gather``; collective counts n-INDEPENDENT past XLA's fixed
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-443-    boundary-iteration peel, i.e. the only collectives are the scan
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:444:    body's in-stage tile-halo ``collective-permute``s and, with
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-445-    ``fix_mass``, the fixer ``psum``).
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-446-    The scanned body is the SAME blocked step — input layout == output
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-447-    layout (tile-sharded), so the carry never leaves tile layout and no
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-448-    D-grid<->cell conversion runs inside the scan (those live in
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-449-    ``enter``/``exit_`` only, once per segment boundary).

exec
/usr/bin/bash -lc "wc -l docs/performance/scaling/levante_campaign_2026-07-24.md && sed -n '1760,2120p' docs/performance/scaling/levante_campaign_2026-07-24.md && rg --files scripts/tmp 2>/dev/null | rg 'probe_latlon_halo_census|26630370|26630438|latlon.*(bound|census)' || true" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 278ms:
1834 docs/performance/scaling/levante_campaign_2026-07-24.md
s8 np8/16/32, lloyd=0, f32, sfc + `--reorder-for 128`, steps 12 /
warmup 3 — configuration-matched to the s9 ladder (26600095); NOT fully
reproduction-grade: the s9 np32 row's git_sha reads `unknown`, and this
run's `7151d12a1-dirty` has no archived dirty-file manifest (the live
diff touched only doc+plot, which supports but cannot retrospectively
prove the bench path was untouched): **6.58 / 6.43 / 7.29 ms**.

Weak pairs (~4x cells with 4x GPUs — global ratio 3.9994 after both
meshes pad +126 cells; tiles NEAR-matched to 0.015 %:
81,936/81,924, 40,968/40,962, 20,484/20,481), SAME lloyd-0 family:

| cells/GPU | s8 rung | s9 rung | ratio | weak eff |
|---|---|---|---|---|
| 81.9k | np8 6.58 | np32 12.47 | 1.895 | **0.53** |
| 41.0k | np16 6.43 | np64 9.60 | 1.493 | 0.67 |
| 20.5k | np32 7.29 | np128 11.48 | 1.575 | 0.64 |

* The falsifiability block's CONFIRM branch fires: ratios stay well
  above 1 with the known Lloyd-family mismatch REMOVED. (This does not
  prove the old confound "only" biased the size — these are
  unreplicated single runs from separate allocations, one comparator
  without row-level provenance; the confounded draft read
  1.80/1.35/1.41 vs 1.90/1.49/1.57 here, and the production-mesh s8
  np8 was 6.92 vs lloyd-0 6.58, -4.9 %, so the mesh family does shift
  absolutes.)
* Restated: at NEAR-matched per-GPU tile, ~quadrupling devices+problem
  costs 1.5-1.9x on this lane — the GPU-side analogue of the ocean CPU
  scale-out term. Weak efficiency 0.53-0.67 at 4x. Mechanism still
  UNATTRIBUTED (PLAUSIBLE candidates unchanged: inter-node neighbour
  fraction growth, collective latency vs count, sfc partition-quality
  decay with parts; the metis receipt argues against pure
  partition-cut explanations, on the CPU lane at least).
* The non-monotone tile dependence of the ratio (largest at the
  LARGEST tile, 1.90 at 81.9k) is unexplained; recorded, not theorised.

## Distance-to-modeled-limit: atm lat-lon GPU (2026-08-02, "near theoretical limit" directive)

Closed the bench's own honest-null bound gap (audit item 4) for the
lat-lon lane, using only repo instruments:

* **Halo census** (new probe `scripts/tmp/_probe_latlon_halo_census.py`,
  virtual-CPU forced-host-platform lowering of the REAL
  `make_sharded_atm_latlon_step`): **41 collective-permutes + 1
  all-reduce per step**, nd-INDEPENDENT (identical at nd=8 and nd=16 —
  the 1-D band structure check). Exact CP payload from compiled-HLO
  result shapes: 4,635,408 B/dev/step at n_lon=1024 L26 f32 = 1.06x the
  single-row slab model; linear in n_lon (checked 1024 vs 2048, 0.07 %
  residual) -> **18.5 MB/dev/step at n_lon=4096 f32**. CAVEAT: CPU
  lowering; GPU-side collective combining could change the executed
  count (metadata.py:214) — the bound is a MODEL.
* **Same-tile nd=1 compute baselines** (job 26630370, roofline recipe):
  16x4096 f32 1.659 ms, 32x4096 f32 2.837, 16x4096 f64 2.973.
  Approximation, recorded: nd=1 includes pole tiles -> compute term
  biased HIGH -> bound conservative.
* **Calibrated bound** (`metadata.calibrated_bound`, measured fabric
  constants: IB 26.3 us / 23.5 GB/s, NVLink 17.8 / 64.2):

| row | measured | t_bound (IB) | measured/bound |
|---|---|---|---|
| LL2048@64 f32 | 6.732 | 2.863 | **2.35** |
| LL2048@128 f32 | 5.577 | 1.848 | **3.02** |
| LL2048@128 f64 | 9.602 | 2.999 | **3.20** |

* **The lat-lon GPU panel sits ~2.4-3.2x ABOVE its modeled limit** — the
  eff-0.60 strong leg is NOT close to the fabric+compute floor. Leading
  PLAUSIBLE mechanism (uninstrumented): effective per-CP cost
  (launch + schedule + stream sync) well above the raw 26 us fabric
  latency across 41 dependency-chained exchanges — the arXiv:2607.16100
  small-collective regime. The model itself notes the serialized-latency
  vs overlap biases pull opposite ways; treat measured/bound as a
  consistency diagnostic, not proven headroom.
* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
  same-job control, falsifiability block in the script. The ocean-lane
  null for these flags was reduction-dominated — first atm test.

exec
/usr/bin/bash -lc "ls -la scripts/tmp | sed -n '1,240p'; find scripts/tmp -maxdepth 1 -type f -name '*latlon*' -printf '%f\\n' 2>/dev/null; rg -n -C 8 \"calibrated_bound|def calibrated_bound|collective-permute|collective_permute|combine-8|pipelined|26630438|26630370|falsif|CONFIRM|REFUTE\" scripts/tmp packages docs/performance/scaling --glob '*.py' --glob '*.md' 2>/dev/null" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 643ms:
total 380
drwxrws---+  3 b381103 bd1083 12288 Aug  2 16:46 .
drwxrws---+ 13 b381103 bd1083  4096 Jul 28 15:44 ..
drwxrws---+  2 b381103 bd1083  4096 Jul 30 10:50 __pycache__
-rw-rw----+  1 b381103 bd1083  1412 Jul 30 10:50 _bufdump_compileonly.py
-rw-rw----+  1 b381103 bd1083  4105 Jul 28 18:38 _memprobe_ocean.py
-rw-rw----+  1 b381103 bd1083  3202 Aug  2 16:52 _probe_latlon_halo_census.py
-rw-rw----+  1 b381103 bd1083  2017 Jul 24 16:50 anchor_1gpu_derecho_match.sbatch
-rw-rw----+  1 b381103 bd1083  2109 Jul 24 21:32 atm_ladders_final.sbatch
-rw-rw----+  1 b381103 bd1083  1586 Jul 25 01:25 comm_microbench.sbatch
-rw-rw----+  1 b381103 bd1083  2165 Jul 28 06:01 cpu1024_matched.sbatch
-rw-rw----+  1 b381103 bd1083  2685 Jul 28 02:00 cpu_1024.sbatch
-rw-rw----+  1 b381103 bd1083  1412 Jul 27 14:12 cpu_s8_np512.sbatch
-rw-rw----+  1 b381103 bd1083  3067 Jul 26 12:24 cpu_wetbal_numa.sbatch
-rw-rw----+  1 b381103 bd1083  1109 Jul 24 18:54 cube_face6_baseline.sbatch
-rw-rw----+  1 b381103 bd1083  1109 Jul 24 18:58 cube_face6_c384.sbatch
-rw-rw----+  1 b381103 bd1083  1386 Jul 24 19:24 cube_fair_c192.sbatch
-rw-rw----+  1 b381103 bd1083  1386 Jul 24 20:37 cube_fair_c768.sbatch
-rw-rw----+  1 b381103 bd1083  1386 Jul 24 19:17 cube_fair_ladder.sbatch
-rw-rw----+  1 b381103 bd1083  1646 Jul 27 14:13 cube_fixedtile_L30.sbatch
-rw-rw----+  1 b381103 bd1083  1878 Jul 27 14:34 cube_fixedtile_small.sbatch
-rw-rw----+  1 b381103 bd1083  2056 Jul 28 03:05 cube_nsys_fixed.sbatch
-rw-rw----+  1 b381103 bd1083  1646 Jul 28 20:46 cube_skew.sbatch
-rw-rw----+  1 b381103 bd1083  1459 Jul 29 09:13 fig3_atm128.sbatch
-rw-rw----+  1 b381103 bd1083  2324 Jul 29 09:13 fig3_cpu_f32.sbatch
-rw-rw----+  1 b381103 bd1083  1849 Jul 29 13:14 fig3_mpas128.sbatch
-rw-rw----+  1 b381103 bd1083  1223 Jul 29 09:13 fig3_mpas32.sbatch
-rw-rw----+  1 b381103 bd1083  1448 Jul 29 09:13 fig3_oc128.sbatch
-rw-rw----+  1 b381103 bd1083  1417 Jul 27 11:50 fig_cube_f64.sbatch
-rw-rw----+  1 b381103 bd1083  1841 Jul 27 11:09 fig_gap_f64.sbatch
-rw-rw----+  1 b381103 bd1083  1323 Jul 27 11:22 fig_ico_f32.sbatch
-rw-rw----+  1 b381103 bd1083  1465 Jul 27 11:53 fig_ico_f64_matched.sbatch
-rw-rw----+  1 b381103 bd1083  1557 Jul 28 06:57 gap_cube_f64_tiled.sbatch
-rw-rw----+  1 b381103 bd1083  1665 Jul 28 06:58 gap_tripole16.sbatch
-rw-rw----+  1 b381103 bd1083   844 Jul 24 18:40 gate_selfspawn.sbatch
-rw-rw----+  1 b381103 bd1083   657 Jul 26 12:26 gather_stencil.sbatch
-rw-rw----+  1 b381103 bd1083  2067 Jul 28 19:54 ll2304_retry.sbatch
-rw-rw----+  1 b381103 bd1083  1568 Jul 28 20:57 memprobe.sbatch
-rw-rw----+  1 b381103 bd1083  2084 Jul 26 13:03 mpas_barrier_ab.sbatch
-rw-rw----+  1 b381103 bd1083  2127 Jul 24 23:11 mpas_codegen_ab.sbatch
-rw-rw----+  1 b381103 bd1083  1622 Jul 27 10:03 mpas_f64_ladder.sbatch
-rw-rw----+  1 b381103 bd1083  1520 Jul 27 10:10 mpas_f64_np8_receipt.sbatch
-rw-rw----+  1 b381103 bd1083   927 Jul 24 22:29 mpas_hlo_diff.sbatch
-rw-rw----+  1 b381103 bd1083  1886 Jul 26 12:52 mpas_np4_fusion_flags.sbatch
-rw-rw----+  1 b381103 bd1083  1996 Jul 26 12:46 mpas_np4_hlo.sbatch
-rw-rw----+  1 b381103 bd1083  1973 Jul 26 12:32 mpas_np4_nsys.sbatch
-rw-rw----+  1 b381103 bd1083  1744 Jul 24 23:01 mpas_partmethod_ab.sbatch
-rw-rw----+  1 b381103 bd1083  1842 Jul 24 21:42 mpas_recheck.sbatch
-rw-rw----+  1 b381103 bd1083  2979 Aug  2 12:44 mpas_s9_ensemble.sbatch
-rw-rw----+  1 b381103 bd1083  2318 Jul 31 19:15 mpas_s9_ladder.sbatch
-rw-rw----+  1 b381103 bd1083  1582 Jul 27 11:09 mpasoc_cpu_ladder.sbatch
-rw-rw----+  1 b381103 bd1083  1946 Jul 27 23:16 mpasoc_fixed_rpn.sbatch
-rw-rw----+  1 b381103 bd1083  3471 Jul 31 19:15 mpasoc_metis_placement.sbatch
-rw-rw----+  1 b381103 bd1083  2400 Jul 27 23:30 mpasoc_partition_2x2.sbatch
-rw-rw----+  1 b381103 bd1083  1705 Jul 27 19:56 mpasoc_scaleout.sbatch
-rw-rw----+  1 b381103 bd1083  1584 Jul 30 02:05 oc128_ab.sbatch
-rw-rw----+  1 b381103 bd1083  1550 Jul 30 10:50 oc128_bufdump.sbatch
-rw-rw----+  1 b381103 bd1083  1576 Jul 24 16:40 ocean_ab_ncclproto.sbatch
-rw-rw----+  1 b381103 bd1083  1758 Jul 24 16:27 ocean_ab_pcg.sbatch
-rw-rw----+  1 b381103 bd1083  2402 Jul 24 15:15 ocean_ab_solver.sbatch
-rw-rw----+  1 b381103 bd1083  1600 Jul 24 18:50 ocean_ab_vmixf32.sbatch
-rw-rw----+  1 b381103 bd1083  1404 Jul 24 16:41 ocean_ab_widefused.sbatch
-rw-rw----+  1 b381103 bd1083  1996 Jul 24 17:53 ocean_ab_xla.sbatch
-rw-rw----+  1 b381103 bd1083  1575 Jul 25 02:24 ocean_calibrated_bound.sbatch
-rw-rw----+  1 b381103 bd1083  1970 Jul 25 06:24 ocean_config_headtohead.sbatch
-rw-rw----+  1 b381103 bd1083  1807 Jul 27 10:17 ocean_cov2.sbatch
-rw-rw----+  1 b381103 bd1083  1915 Jul 27 10:04 ocean_coverage_lanes.sbatch
-rw-rw----+  1 b381103 bd1083  1766 Jul 25 06:40 ocean_crossover_f32_nd4.sbatch
-rw-rw----+  1 b381103 bd1083  1615 Jul 28 03:06 ocean_fixedtile32.sbatch
-rw-rw----+  1 b381103 bd1083  1763 Jul 25 02:29 ocean_full_roofline.sbatch
-rw-rw----+  1 b381103 bd1083  1804 Jul 25 05:16 ocean_iter_compute_check.sbatch
-rw-rw----+  1 b381103 bd1083  2053 Jul 27 10:03 ocean_mixed_precision.sbatch
-rw-rw----+  1 b381103 bd1083  2224 Jul 25 04:04 ocean_overlap_test.sbatch
-rw-rw----+  1 b381103 bd1083  1851 Jul 25 04:41 ocean_pcg_iters_sweep.sbatch
-rw-rw----+  1 b381103 bd1083  2583 Jul 25 02:34 ocean_roofline_correct.sbatch
-rw-rw----+  1 b381103 bd1083  1906 Jul 28 01:30 ocean_scaleout64.sbatch
-rw-rw----+  1 b381103 bd1083  1958 Jul 25 05:49 ocean_singlereduce_slope.sbatch
-rw-rw----+  1 b381103 bd1083  1908 Jul 25 12:36 ocean_stability_repeats.sbatch
-rw-rw----+  1 b381103 bd1083  1586 Jul 24 20:11 ocean_weak_prod.sbatch
-rw-rw----+  1 b381103 bd1083  2004 Jul 25 07:12 ocean_widehalo_stability.sbatch
-rw-rw----+  1 b381103 bd1083  7431 Jul 24 15:25 plot_campaign_cpu_mpi.py
-rw-rw----+  1 b381103 bd1083   507 Jul 30 01:04 prewarm_s9.sbatch
-rw-r-----+  1 b381103 bd1083  2394 Jul 25 03:11 probe_ocean_halo_bytes.py
-rw-r-----+  1 b381103 bd1083  2757 Jul 25 03:39 probe_sharded_overhead.py
-rw-rw----+  1 b381103 bd1083  1972 Jul 28 10:04 s9_insurance.sbatch
-rw-rw----+  1 b381103 bd1083  1839 Jul 27 12:30 scaleout_cpu_big.sbatch
-rw-rw----+  1 b381103 bd1083  2153 Jul 27 13:52 scaleout_cube_kt3.sbatch
-rw-rw----+  1 b381103 bd1083  2424 Jul 27 12:32 scaleout_cube_matched.sbatch
-rw-rw----+  1 b381103 bd1083  2010 Jul 27 12:43 scaleout_cube_tiled96.sbatch
-rw-rw----+  1 b381103 bd1083  2001 Jul 27 17:47 scaleout_latlon64.sbatch
-rw-rw----+  1 b381103 bd1083   730 Jul 25 03:35 sharded_overhead.sbatch
-rw-rw----+  1 b381103 bd1083  1311 Jul 26 13:26 wetbal_only.sbatch
_probe_latlon_halo_census.py
scaleout_latlon64.sbatch
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-173-  the old '67% fallback artifact' caveat is CLOSED, the cost is real
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-174-  Thomas-solve bandwidth; ts_pair 1.58x isolated vs two solves; CPU
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-175-  vmix is at its bandwidth floor post-pair.
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-176-- ocean multi-node: np16 across 2 nodes LL192 = 90.2 ms vs np8 138 =
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-177-  near-ideal 8->16 (1-node np16 flattening WAS per-node DRAM
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-178-  contention); np32@16/node = 123 ms (slower than np16@8/node) =>
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-179-  ranks/node <= 8 policy at LL192; weak np32 growth x7.9 (vs x2.95
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-180-  np8) => cross-node allreduce/sendrecv latency reactivates the
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md:181:  PCG reduction-count lever (pipelined-CG / batched dots) at >=2 nodes.
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-182-- post-lever ocean phase split np4 LL128 (78.7 ms): vmix 39.7%,
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-183-  baroclinic 20.1%, barotropic 9.5%, tracer 9.6%.
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-184-
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-185-Shipped this round (codex-reviewed, gates job 8460429 green):
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-186-- ATM lat-lon fused entry pads: one pad_with_pole_bc_lat_multi((T, u))
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-187-  per tendency eval; padded u SHARED by curl circulation + absolute-
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-188-  vorticity 4-pt average (was padded twice).  Census 15 -> 9
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md-189-  exchanges/step (+12 already constant-folded for free from the ocean
--
docs/performance/scaling/SCALING_STATUS_AUDIT.md-20-that qualifier is wrong.
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
docs/performance/scaling/SCALING_STATUS_AUDIT.md-36-| **tripole / eORCA** | (c) fold-aware band-MPI validated at operator level AND full-model (#883: full-model tripole MPI validation + tripole bench lanes on `bench_ocean_mpi_scaling.py`) | (c) tripole fold wired into the SPMD step (#883: `make_sharded_ocean_step` fold support; `bench_ocean_latlon_spmd_scaling.py --tripole`); single-GPU throughput (b: 313 Mc/s fp32, `SCALING_SUMMARY.md`); no multi-device scale receipts | none | scale receipts only | tripole rank/device ladders on Derecho/Levante (lanes exist) |
--
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-33-### Ocean
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-34-
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-35-| Model | Machine / devices | Config | Headline | Per-A100 fp64 throughput |
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-36-|---|---|---|---|---|
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-37-| Oceananigans (Silvestri 2025, JAMES 2024MS004465) | Perlmutter, 64 A100 | 1/12°, 100L | 10 SYPD | **~160–230 Mcell-updates/s** (derived) |
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-38-| Oceananigans | 16 A100 | 1/4° | ~75 SYPD | — |
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-39-| Oceananigans weak | 4→768 A100 | 5e7–2e8 cells/GPU | **~1.0 eff (flat)** | — |
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-40-| Oceananigans strong | 4×→16× GPUs | 1/48° | ideal→0.70 (land-imbalance, not comm) | — |
docs/performance/scaling/derecho_levante_sota_review_2026-07.md:41:| MPAS-O (Kang 2021) | Cori-KNL 16,320 cores | EC60to30 | semi-implicit pipelined BiCGStab+RAS: barotropic 2.9×, model 1.9× vs split-explicit subcycling | — |
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-42-| Omega v0.1 (2026) | Frontier | — | device-resident packed halos + GPU-aware MPI = **4–6× lower halo time** vs host-staged | — |
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-43-| ClimaOcean (Wagner 2025) | 1 H100 | 16 km coupled | 1.5 SYPD | — |
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-44-
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-45-legoESM ocean anchors: LL192 fp64 full step ~152 Mcells/s on a crippled-fp64
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-46-laptop GPU (fp64=1/64) — the A100 number is the first thing to measure with the
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-47-new `ocean_gpu_scaling` jobs; the architecture (band decomposition,
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-48-split-explicit option with local clamp, fused halos, mixed-precision opt-ins)
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-49-is aligned with what makes Oceananigans fast. Their two defining techniques we
--
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-76-- XLA latency-hiding scheduler ON by default in `runtime/backend.py` (their #1 flag).
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-77-- Whole-step jit ≡ Pace's DaCe orchestration ≡ ICON's DaCe win (we get it natively).
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-78-- Halo aggregation: `pad_halo_4d` + multi-field stacking + dtype-cast fusion ≡
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-79-  Omega's packed device-resident exchanges; MPAS batched union-neighbor halo default.
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-80-- SFC/METIS Voronoi partition (`voronoi_partition.py method="auto"/"sfc"`) ≡
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-81-  ClimaCore's space-filling-curve element partition.
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-82-- Barotropic: split-explicit (halo-only) + reduction-free local clamp opt-in ≡
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-83-  the Oceananigans no-global-collectives doctrine; single_reduce (ChronGear) +
docs/performance/scaling/derecho_levante_sota_review_2026-07.md:84:  P-CSI/Chebyshev already characterized; MPAS-O's pipelined-BiCGStab+RAS is the
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-85-  known alternative if implicit-CN must scale to high ranks.
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-86-
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-87-## 3. Gaps found in THIS review and closed this session
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-88-
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-89-1. **Ocean SPMD full-step retrace bug (real per-step recompile)** —
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-90-   `make_sharded_ocean_step` rebuilt an un-jitted `shard_map` EVERY call
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-91-   (the exact bug fixed on the atm twin in `fa2ce32b6`: 142 s/step → 1.1 ms).
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-92-   Fixed: jit-once cache + `dt` as traced operand
--
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-238-   the multi-controller variant needs the shard-per-process state constructor.
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-239-4. **Tripole/eORCA active fold under SPMD** (`sharded_ocean_step` raises;
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-240-   MPI band path already supports the fold) — needed for eORCA GPU scaling.
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-241-5. **MPAS-ocean scaling lane** (voronoi_mpi step exists; no bench drives it).
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-242-6. **Cube np>6 sub-face production step** (P4): transport+KE stack tiled and
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-243-   parity-proven; d_sw1/d_sw5/d_sw6 assembly remains — THE lever for >6 GPUs
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-244-   on cubed-sphere, now benchable on Derecho/Levante once assembled.
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-245-7. **Pipelined-p2p XLA experiment** for scan-loop ppermutes
docs/performance/scaling/derecho_levante_sota_review_2026-07.md:246:   (`--xla_gpu_enable_pipelined_p2p` + permute decomposer; command buffers off)
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-247-   — experimental flag, A/B only.
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-248-
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-249-## 4b. Codex adversarial review
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-250-
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-251-3 rounds (thread 019f25ce-1281): round 1 = 7 findings (1 rejected with
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-252-receipt — the smoke IS above the 2-row floor and ran green); fixes in
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-253-`d760e48cc` (structure-keyed jit cache ocean+atm, loud MPI failures,
docs/performance/scaling/derecho_levante_sota_review_2026-07.md-254-PALS guard, indexed GPU pinning) and `b9c43586d` (env-routed
--
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-49-multicontroller, 8 GPU), lane D (ocean multicontroller, 8 GPU), lane E
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-50-(icosahedral/MPAS multicontroller, 6 GPU). Larger ladders: resubmit with
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-51-more nodes and `GPU_RANKS`/`LL_RES` etc. per the README.
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-52-
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-53-**Production knob (measured 2026-07-09, use it):** add
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-54-`JAX_ENABLE_PGLE=true JAX_PGLE_PROFILING_RUNS=3` to the environment of
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-55-lat-lon-class route-B lanes — measured **+8.5%** (5.97 vs 6.48 ms/step at
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-56-8 GPU). Do **not** set `LEGOESM_LATLON_SPMD_FUSED_HALO=1` (measured −12%
docs/performance/scaling/derecho_colleague_runbook_2026-07.md:57:on latlon at this size) and do not add the CP-combine/pipelined-p2p XLA
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-58-flags (measured −10% combined). Full A/B table:
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-59-`docs/performance/scaling/spmd_message_census_2026-07-08.md`.
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-60-
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-61-**Resolution floors (or the curves will look "broken" and are not):**
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-62-keep ≥ ~30k columns per GPU. Concretely: lat-lon LL512+ and icosahedral
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-63-L7+ for 8–16 GPUs; coarse grids (156 km latlon, 112 km ico, C48 cube) do
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-64-NOT strong-scale on GPUs anywhere — per-device saturation, expected.
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-65-
--
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-9-- **XLA latency-hiding scheduler** (#3): `xla_gpu_enable_latency_hiding_
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-10-  scheduler=true` is ALREADY in runtime/backend.py NVIDIA_GPU_XLA_FLAGS
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-11-  (line 192) + `xla_gpu_enable_highest_priority_async_stream=true`. The
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-12-  agent's "OFF by default" is the XLA default; our backend overrides it.
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-13-  Explains the campaign note "XLA already overlaps GPU collectives".
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-14-- **XLA command-buffer / CUDA-graph capture** (#4): ALREADY set
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-15-  `xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL,COLLECTIVES` (line 199)
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-16-  — CUSTOM_CALL is included, so the LAPACK gtsv FFI IS in the captured set
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md:17:  (no graph fragmentation from it). (Worth a one-off Nsight trace to CONFIRM
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-18-  the FV3 step is one graph, but the flag coverage is already correct.)
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-19-- **Pipelined/communication-avoiding Krylov** (#5): equivalent SHIPPED
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-20-  (Chronopoulos-Gear single_reduce + Chebyshev reduction-free precond). The
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-21-  2025 POP/CG-variant papers beat a naive PCG baseline we're already past.
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-22-- **Ocean f32 path** (part of #2): ALREADY clean + measured 1.72x LL128
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-23-  (campaign L2); production runs f64 by SCIENTIFIC choice, not a missing
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-24-  lever. Tensor-core/TF32/bf16: future-hardware (Turing has none; gated).
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md-25-- **Distributed SHT / JAX-Fluids 2.0 SPMD / Shardy**: same SPMD class we
--
docs/performance/scaling/scaling_review_2026-06-13.md-44-   weak-scaling gap; then stack same-stage fields → 1 exchange. Needs the
docs/performance/scaling/scaling_review_2026-06-13.md-45-   `batched_halo_exchange` 18× CPU regression fixed first (pack via scatter →
docs/performance/scaling/scaling_review_2026-06-13.md-46-   gather).
docs/performance/scaling/scaling_review_2026-06-13.md-47-7. **RAS / block-Jacobi-1-halo preconditioner** (ocean) — if Chebyshev's
docs/performance/scaling/scaling_review_2026-06-13.md-48-   iteration count climbs with rank count (it won't much at ≤32 ranks).
docs/performance/scaling/scaling_review_2026-06-13.md-49-8. **Fused K-build into Thomas (5a)** + **CPU tridiagonal → LAPACK `gtsv`**
docs/performance/scaling/scaling_review_2026-06-13.md-50-   (lax.linalg.tridiagonal_solve is now LAPACK-batched + has a JVP; the legacy
docs/performance/scaling/scaling_review_2026-06-13.md-51-   fori-loop is ~2000× slower). Incremental, AD-safe.
docs/performance/scaling/scaling_review_2026-06-13.md:52:9. **XLA flags**: `xla_gpu_collective_permute_decomposer_threshold` +
docs/performance/scaling/scaling_review_2026-06-13.md-53-   all_reduce/all_gather combine thresholds (absent) — overlap interior compute
docs/performance/scaling/scaling_review_2026-06-13.md-54-   with ppermute halos + merge small collectives. A/B-gated (low-impact on
docs/performance/scaling/scaling_review_2026-06-13.md-55-   PCIe, future-proofs NVLink + cuts Gloo message count).
docs/performance/scaling/scaling_review_2026-06-13.md-56-
docs/performance/scaling/scaling_review_2026-06-13.md-57-### Tier 3 — big / deliberate
docs/performance/scaling/scaling_review_2026-06-13.md-58-10. **IMEX semi-implicit extended to the ocean baroclinic** + default-on the
docs/performance/scaling/scaling_review_2026-06-13.md-59-    spectral SI (machinery exists in semi_implicit.py) — larger dt, fewer
docs/performance/scaling/scaling_review_2026-06-13.md-60-    barotropic subcycles. Multi-week, validate.
--
docs/performance/scaling/scaling_crm_gpu.md-134-- HBM utilization at default: ~63% → ~95% sustained
docs/performance/scaling/scaling_crm_gpu.md-135-- fp64 PCR: +51% over cuSPARSE (ALU-bound, not memory-bound)
docs/performance/scaling/scaling_crm_gpu.md-136-
docs/performance/scaling/scaling_crm_gpu.md-137-**Why this is "as close as possible to theoretical":**
docs/performance/scaling/scaling_crm_gpu.md-138-1. At default config we're at 95% HBM (upper-bound estimate; lower
docs/performance/scaling/scaling_crm_gpu.md-139-   bound ~30% — true value somewhere between, depending on XLA L2
docs/performance/scaling/scaling_crm_gpu.md-140-   reuse). Memory bandwidth IS the bottleneck.
docs/performance/scaling/scaling_crm_gpu.md-141-2. 135 GPU kernels/step (HLO-counted). With 5 us launch each, ~675us
docs/performance/scaling/scaling_crm_gpu.md:142:   is launch overhead — but most kernels run concurrently/pipelined
docs/performance/scaling/scaling_crm_gpu.md-143-   on modern NVIDIA HW.
docs/performance/scaling/scaling_crm_gpu.md-144-3. PCR replaces cuSPARSE custom_call → eliminates the only
docs/performance/scaling/scaling_crm_gpu.md-145-   fusion barrier in the substep.
docs/performance/scaling/scaling_crm_gpu.md-146-4. All 4 NH dycores benefit via shared `acoustic_substeps_semi_implicit`.
docs/performance/scaling/scaling_crm_gpu.md-147-
docs/performance/scaling/scaling_crm_gpu.md-148-**Remaining headroom (NOT pursued — outside "minimum code" scope):**
docs/performance/scaling/scaling_crm_gpu.md-149-- Custom CUDA kernel for PCR with cooperative thread-block reduction:
docs/performance/scaling/scaling_crm_gpu.md-150-  could merge 5 PCR levels into 1 kernel, gain ~5-10%.
--
docs/performance/scaling/spmd_message_census_2026-07-08.md-16-message count × per-message latency**. This note pins the actual counts.
docs/performance/scaling/spmd_message_census_2026-07-08.md-17-
docs/performance/scaling/spmd_message_census_2026-07-08.md-18-## Method
docs/performance/scaling/spmd_message_census_2026-07-08.md-19-
docs/performance/scaling/spmd_message_census_2026-07-08.md-20-Optimized-HLO census on CPU virtual devices (message *count* is a static
docs/performance/scaling/spmd_message_census_2026-07-08.md-21-schedule — resolution-independent — so a C24/L8 or 64×128/L8 compile gives
docs/performance/scaling/spmd_message_census_2026-07-08.md-22-the production count). One-off probes (untracked scratch, `scripts/tmp/`);
docs/performance/scaling/spmd_message_census_2026-07-08.md-23-the method is three lines and the cube tiled bench already records it
docs/performance/scaling/spmd_message_census_2026-07-08.md:24:per-row as `hlo_collective_permutes`:
docs/performance/scaling/spmd_message_census_2026-07-08.md-25-
docs/performance/scaling/spmd_message_census_2026-07-08.md-26-```python
docs/performance/scaling/spmd_message_census_2026-07-08.md-27-# XLA_FLAGS=--xla_force_host_platform_device_count=N  JAX_PLATFORMS=cpu
docs/performance/scaling/spmd_message_census_2026-07-08.md-28-hlo = jit_step.lower(state, dt).compile().as_text()
docs/performance/scaling/spmd_message_census_2026-07-08.md:29:n_cp = hlo.count("collective-permute-start") + hlo.count("collective-permute(")
docs/performance/scaling/spmd_message_census_2026-07-08.md-30-```
docs/performance/scaling/spmd_message_census_2026-07-08.md-31-
docs/performance/scaling/spmd_message_census_2026-07-08.md-32-Counted alongside `all-gather` / `all-reduce` occurrences.
docs/performance/scaling/spmd_message_census_2026-07-08.md-33-
docs/performance/scaling/spmd_message_census_2026-07-08.md-34-## Results
docs/performance/scaling/spmd_message_census_2026-07-08.md-35-
docs/performance/scaling/spmd_message_census_2026-07-08.md-36-### Cube cs-spmd full PE step (C24/L8 shape; counts are shape-independent)
docs/performance/scaling/spmd_message_census_2026-07-08.md-37-
docs/performance/scaling/spmd_message_census_2026-07-08.md:38:| devices | collective-permutes / step | all-reduce / step |
docs/performance/scaling/spmd_message_census_2026-07-08.md-39-|---|---|---|
docs/performance/scaling/spmd_message_census_2026-07-08.md-40-| 2 | 15 | 1 |
docs/performance/scaling/spmd_message_census_2026-07-08.md-41-| 3 | 26 | 1 |
docs/performance/scaling/spmd_message_census_2026-07-08.md-42-| 6 | 46 | 1 |
docs/performance/scaling/spmd_message_census_2026-07-08.md-43-
docs/performance/scaling/spmd_message_census_2026-07-08.md-44-The single all-reduce is the conservation fixer (fine). The 46 CPs at 6
docs/performance/scaling/spmd_message_census_2026-07-08.md-45-devices = ~11.5 exchange points × 4 ppermute rounds (the edge-coloring
docs/performance/scaling/spmd_message_census_2026-07-08.md-46-matching floor: each face has 4 neighbors, so 4 rounds is the minimum for
--
docs/performance/scaling/spmd_message_census_2026-07-08.md-58-
docs/performance/scaling/spmd_message_census_2026-07-08.md-59-At C48/L26 a 1-A100 step is ~2.6 ms; 46 sequential small CPs at
docs/performance/scaling/spmd_message_census_2026-07-08.md-60-~30–80 µs NCCL p2p latency ≈ 1.5–3.5 ms — that IS the anti-scaling of the
docs/performance/scaling/spmd_message_census_2026-07-08.md-61-coarse cube curves, and the 2-GPU dip (15 CPs but 3 faces/shard + sync per
docs/performance/scaling/spmd_message_census_2026-07-08.md-62-CP on a halved compute slice).
docs/performance/scaling/spmd_message_census_2026-07-08.md-63-
docs/performance/scaling/spmd_message_census_2026-07-08.md-64-### Atm latlon SPMD step (64×128/L8, 4 devices)
docs/performance/scaling/spmd_message_census_2026-07-08.md-65-
docs/performance/scaling/spmd_message_census_2026-07-08.md:66:| arm | collective-permutes / step |
docs/performance/scaling/spmd_message_census_2026-07-08.md-67-|---|---|
docs/performance/scaling/spmd_message_census_2026-07-08.md-68-| baseline | 41 |
docs/performance/scaling/spmd_message_census_2026-07-08.md-69-| `LEGOESM_LATLON_SPMD_FUSED_HALO=1` | **29 (−29%)** |
docs/performance/scaling/spmd_message_census_2026-07-08.md-70-
docs/performance/scaling/spmd_message_census_2026-07-08.md-71-The fused multi-pad (audit item 7, shipped opt-in for the ocean with a
docs/performance/scaling/spmd_message_census_2026-07-08.md-72-25%-fewer-ppermutes receipt) also covers the atm step via
docs/performance/scaling/spmd_message_census_2026-07-08.md-73-`pad_with_pole_bc_lat_multi` — same bit-identical packing, same flag.
docs/performance/scaling/spmd_message_census_2026-07-08.md-74-Remaining 29 CPs live inside shared per-operator pads
--
docs/performance/scaling/spmd_message_census_2026-07-08.md-79-
docs/performance/scaling/spmd_message_census_2026-07-08.md-80-## MEASURED lane-T verdicts (Derecho, 2026-07-09, 8×A100 route-B NCCL,
docs/performance/scaling/spmd_message_census_2026-07-08.md-81-## same-allocation A/B — latlon LL512-class atm, ocean LL288-class)
docs/performance/scaling/spmd_message_census_2026-07-08.md-82-
docs/performance/scaling/spmd_message_census_2026-07-08.md-83-| arm | latlon ms/step | latlon SYPD | ocean ms/step | verdict |
docs/performance/scaling/spmd_message_census_2026-07-08.md-84-|---|---|---|---|---|
docs/performance/scaling/spmd_message_census_2026-07-08.md-85-| base | 6.48 | 25.4 | 33.09 | control |
docs/performance/scaling/spmd_message_census_2026-07-08.md-86-| fused (`LEGOESM_LATLON_SPMD_FUSED_HALO=1`) | 7.36 | 22.3 | 32.72 | **latlon −12 % — default stays OFF**; ocean +1 % (noise) |
docs/performance/scaling/spmd_message_census_2026-07-08.md:87:| xla (CP-combine 32 MiB + pipelined p2p) | 7.22 | 22.8 | — | **−10 % — not recommended as-is**; split the two flags in a follow-up arm before discarding |
docs/performance/scaling/spmd_message_census_2026-07-08.md-88-| pgle (`JAX_ENABLE_PGLE=true`) | **5.97** | **27.5** | — | **+8.5 % — the winner**; recommend per-run on route-B latlon lanes |
docs/performance/scaling/spmd_message_census_2026-07-08.md-89-
docs/performance/scaling/spmd_message_census_2026-07-08.md-90-Readings:
docs/performance/scaling/spmd_message_census_2026-07-08.md-91-1. **Fused multi-pad loses at this size/count**: −29 % messages, but each
docs/performance/scaling/spmd_message_census_2026-07-08.md-92-   message ~4× larger plus the pack/unpack concats — at LL512/np8 the
docs/performance/scaling/spmd_message_census_2026-07-08.md-93-   per-message latency saved is smaller than the copy overhead added.
docs/performance/scaling/spmd_message_census_2026-07-08.md-94-   The flag stays opt-in (it may still win at higher rank counts /
docs/performance/scaling/spmd_message_census_2026-07-08.md-95-   smaller per-rank tiles where latency dominates — re-A/B there before
docs/performance/scaling/spmd_message_census_2026-07-08.md-96-   discarding).
docs/performance/scaling/spmd_message_census_2026-07-08.md-97-2. **PGLE's profile-guided re-scheduling is the real overlap win** —
docs/performance/scaling/spmd_message_census_2026-07-08.md-98-   +8.5 % without touching the model.  Keep it per-run opt-in
docs/performance/scaling/spmd_message_census_2026-07-08.md-99-   (recompiles after the profiling runs; AOT-incompatible), and wire it
docs/performance/scaling/spmd_message_census_2026-07-08.md-100-   into the production route-B job env for latlon-class lanes.
docs/performance/scaling/spmd_message_census_2026-07-08.md:101:3. **The combined xla arm hurt**; pipelined-p2p and the CP-combiner need
docs/performance/scaling/spmd_message_census_2026-07-08.md-102-   separate arms to attribute (follow-up lane-T variant).
docs/performance/scaling/spmd_message_census_2026-07-08.md-103-4. Caveats: one size per component, one repeat — treat sub-5 % deltas as
docs/performance/scaling/spmd_message_census_2026-07-08.md-104-   noise; the cube base/xla arms and np16/24 rungs are still pending.
docs/performance/scaling/spmd_message_census_2026-07-08.md-105-
docs/performance/scaling/spmd_message_census_2026-07-08.md-106-## Consequences — what to run next on Derecho/Levante
docs/performance/scaling/spmd_message_census_2026-07-08.md-107-
docs/performance/scaling/spmd_message_census_2026-07-08.md-108-The count floor being (near-)reached in code moves the lever to the GPU
docs/performance/scaling/spmd_message_census_2026-07-08.md-109-runtime, in this order (now wired as **lane T** in
docs/performance/scaling/spmd_message_census_2026-07-08.md-110-`gpu_multinode_scaling.pbs` / the Levante twin, outputs under `_ab_tuning/`
docs/performance/scaling/spmd_message_census_2026-07-08.md-111-which the aggregator deliberately skips):
docs/performance/scaling/spmd_message_census_2026-07-08.md-112-
docs/performance/scaling/spmd_message_census_2026-07-08.md-113-1. **Fused-halo flag A/B** (`LEGOESM_LATLON_SPMD_FUSED_HALO=1`) on the
docs/performance/scaling/spmd_message_census_2026-07-08.md-114-   latlon atm + ocean lanes — flip the default per the audit contract only
docs/performance/scaling/spmd_message_census_2026-07-08.md-115-   with this receipt.
docs/performance/scaling/spmd_message_census_2026-07-08.md:116:2. **XLA collective combining + pipelined p2p**
docs/performance/scaling/spmd_message_census_2026-07-08.md:117:   (`--xla_gpu_collective_permute_combine_threshold_bytes`,
docs/performance/scaling/spmd_message_census_2026-07-08.md:118:   `--xla_gpu_enable_pipelined_p2p`): the cube's 4 rounds/exchange are
docs/performance/scaling/spmd_message_census_2026-07-08.md-119-   data-independent — the GPU CollectivePermute combiner can merge
docs/performance/scaling/spmd_message_census_2026-07-08.md-120-   same-round CPs the SPMD partitioner emits separately; pipelining
docs/performance/scaling/spmd_message_census_2026-07-08.md-121-   overlaps them with compute. Code-side round count cannot go below 4
docs/performance/scaling/spmd_message_census_2026-07-08.md-122-   (matching floor), so this is where the cube ≤6-GPU curves have their
docs/performance/scaling/spmd_message_census_2026-07-08.md-123-   remaining headroom.
docs/performance/scaling/spmd_message_census_2026-07-08.md-124-3. **PGLE** (`JAX_ENABLE_PGLE=true`, profiling runs=3): profile-guided
docs/performance/scaling/spmd_message_census_2026-07-08.md-125-   latency estimates re-schedule collectives; not defaulted (recompiles
docs/performance/scaling/spmd_message_census_2026-07-08.md-126-   mid-job, AOT-incompatible).
--
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-3-Scope: consolidate the measured weak/strong scaling bottlenecks for atmosphere
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-4-and ocean on both transports (route-A mpi4jax, route-B NCCL/SPMD), pin the
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-5-LATENCY-bound signal with fresh numbers, and close the two instrument gaps that
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-6-kept that signal off the actual Derecho/Levante runs. Read the campaign context
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-7-first: `derecho_levante_sota_review_2026-07.md` (measured baselines + SOTA),
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-8-`spmd_message_census_2026-07-08.md` (the message-count analysis this extends),
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-9-`SCALING_STATUS_AUDIT.md` (support matrix).
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-10-
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:11:Every claim below is labelled **CONFIRMED** (measured / read from code) or
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-12-**PLAUSIBLE** (inferred, needs a machine receipt).
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-13-
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-14----
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-15-
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-16-## 1. The bottleneck table (what limits each axis, and why)
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-17-
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-18-| Component × axis | Limiter | Evidence | Tier |
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-19-|---|---|---|---|
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:20:| Atm cube strong (≤6 GPU) | **collective-permute COUNT** grows with shard count while per-msg NCCL p2p latency (~30–80 µs) doesn't amortise on small tiles → anti-scales | census: 12 CP @2dev, 41 @6dev (C24/L8, below); f64≡f32 curves ⇒ latency-bound | CONFIRMED |
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:21:| Atm latlon strong | 4→8 GPU node-crossing plateau on route-A (mpi4jax sendrecv OPAQUE to XLA latency-hiding scheduler ⇒ no comm/compute overlap) | Derecho 78 km throughput 600→480→recover@16; route-B multicontroller SHIPPED to remove it | CONFIRMED |
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:22:| Atm ico strong | best-scaling grid (low perimeter/area cell partition); multihost SPMD still open | Derecho 28 km ico eff ~0.38 @16 A100, still rising | CONFIRMED |
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:23:| Atm/all coarse | per-device saturation floor (<~30k cols/GPU flat/anti) — NOT a defect | 111 km latlon/ico FLAT 1→16 A100 | CONFIRMED |
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:24:| Ocean strong (CPU-MPI) | **implicit-CN PCG reduction wall**: default `pcg_variant="standard"` = 2 *sequentially-dependent* all-reduces/iter × `fixed_iters=60` ⇒ ~120 latency-serialized all-reduces/step | `barotropic_common.py` `_fixed_iteration_pcg` (p·Ap @:619 → dependent r·z @:626); np16→32 eff 0.55 | CONFIRMED |
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:25:| Ocean strong (mitigation) | `single_reduce` (Chronopoulos–Gear) → 1 all-reduce/iter; split-explicit + `barotropic_local_subcycle_clamp` → 3 all-reduce/step (reduction-free subcycle) | both exist, both NON-default; beats standard at ≥2 nodes 1.15–1.65× | CONFIRMED |
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:26:| Ocean weak | ~1.0 eff (as SOTA); the strong ceiling is naive land imbalance, not comm | 0.97 weak @590k cells/rank | CONFIRMED |
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:27:| Spectral | single-device by design (both multi-device schemes measured anti-scaling) | prior campaign | CONFIRMED |
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-28-
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-29-Two structural hot-loop items found in the code map (both PLAUSIBLE as
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-30-scaling costs at high step counts, neither a hot-loop collective):
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-31-- **Per-step host dispatch** in the operator-split SPMD driver
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-32-  (`model_driver.py:7783` Python `for` over `seg_steps`) — a compiled
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-33-  `lax.scan`-per-segment path exists (M3b) but the operator-split loop dispatches
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-34-  per step; host-dispatch overhead scales with step count.
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-35-- **Blocking all-gather + full replicated allocation once per sim-day**
--
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-42-coincide ⇒ the multi-GPU leg is **latency-bound, not bandwidth-bound**, so the
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-43-lever is the per-step *message count*, not the byte volume. The count is a
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-44-STATIC compile property — a CPU virtual-device compile yields the same count the
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-45-GPU executes.
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-46-
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-47-Fresh census (sharded cube full PE step, `run_scaling_diagnosis.py --mode
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-48-census`, C24/L8, CPU virtual devices):
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-49-
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:50:| devices | collective-permute / step | all-reduce / step |
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-51-|---|---|---|
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-52-| 2 | 12 | 1 |
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-53-| 6 | 41 | 1 |
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-54-
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:55:Readings (CONFIRMED): the single all-reduce is the conservation fixer (fine).
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-56-The permute count scales ~linearly with the shard count (edge-coloring: each
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-57-face has 4 neighbours ⇒ 4 ppermute rounds × the exchange points) — this IS the
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-58-cube anti-scaling. (The absolute counts are lower than the 15/46 in the
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-59-2026-07-08 note — halo packing improved since — which is exactly why the count
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-60-now belongs on every row as a REGRESSION-tracked metric, not a scratch probe.)
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-61-
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-62-## 3. Instrument gaps closed this session
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-63-
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-64-The diagnosis was well-characterised in the docs, but two gaps kept the signal
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-65-off the real machine runs:
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-66-
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:67:1. **The message census counted only collective-permutes.** The ocean PCG
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-68-   all-reduce wall — the #1 ocean strong-scaling limiter — was invisible to a
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-69-   permute-only census. Fixed: canonical `metadata.count_collectives()` /
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-70-   `hlo_collective_census()` count **all** families (permute + all-reduce +
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-71-   all-gather + all-to-all + reduce-scatter) with the same op-call-form
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-72-   discipline (config-header flag echoes never inflate; async `-start` counted
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:73:   once). The permute family stays bit-identical to `count_collective_permutes`.
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-74-2. **The bottleneck tool ran on NO Derecho/Levante job.** The cluster campaign
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-75-   invoked only the throughput benches (SYPD + permute census); no per-phase
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-76-   halo/reduction/roofline/overlap breakdown was ever captured on the target
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-77-   machines. Fixed:
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-78-   - `run_scaling_diagnosis.py` gained a `census` mode + a census phase in
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-79-     `full`/`quick` (recorded in `summary.json`) — the count is now first-class.
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-80-   - `scripts/cluster/scaling_derecho/diagnosis.pbs` +
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-81-     `scripts/cluster/scaling_levante/diagnosis.sbatch` run the tool across the
--
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-93-   size (C96/L40, ≥30k cols/GPU). Yields the census ladder + per-phase halo BW,
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-94-   reduction latency, roofline, overlap — the WHERE, not just the SYPD.
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-95-2. **Ocean reduction-wall A/B** (the highest-value code lever, already
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-96-   selectable): re-run `bench_ocean_mpi_scaling.py` / the SPMD ocean bench with
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-97-   `--pcg-variant single_reduce` and with split-explicit +
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-98-   `barotropic_local_subcycle_clamp` at ≥16 ranks — the calculus flips toward
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-99-   the reduction-free path where per-message latency dominates.
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-100-3. **Lane-T GPU-runtime A/B** (already wired, `RUN_TUNE=1`): PGLE was the
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:101:   measured winner (+8.5 %); the XLA collective-permute-combine + pipelined-p2p
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-102-   arms still need SPLITTING to attribute (the combined arm hurt −10 %).
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-103-4. Per §1, cube >6 GPU needs the sub-face tiled production step (separate
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-104-   project); the count floor is otherwise reached in code.
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-105-
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-106-## 5. Verification
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-107-
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-108-- `count_collectives` / `hlo_collective_census`:
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-109-  `tests/bench/test_scaling_metadata.py` (all-family synthetic HLO + error-safe
--
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md-20-weak-scaling-capped by **M≈60 iterations × ~111 µs allreduce LATENCY per
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md-21-step**. The single-reduce (Chronopoulos–Gear) lever attacked the *per-
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md-22-iteration reduction count* (3→1) and gave only 3–7% — we marked it
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md-23-"neutralized." **The literature says the real lever is the ITERATION
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md-24-COUNT, not the per-iteration reduction count:**
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md-25-
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md-26-- **MPAS-O** (Kang et al. 2021, JAMES): replaced 30–60 explicit
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md-27-  barotropic subcycles (each a latency-bound halo) with a **semi-implicit
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md:28:  Helmholtz SSH solve** using **pipelined PBiCGStab (2 overlapped
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md-29-  allreduces/iter) + Restricted Additive Schwarz (RAS) preconditioner →
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md-30-  ~6 FIXED iterations** independent of core count. Result: **2.9×
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md-31-  barotropic, 1.9× total at 16,320 cores.** RAS = block-Jacobi (local,
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md-32-  zero-comm) extended by ONE halo layer → 1 halo exchange/precond-apply;
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md-33-  matrix is time-independent so it's assembled+factored once.
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md-34-- **Veros** (Häfner et al. 2021, JAMES): never built a distributed JAX
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md-35-  Krylov at all — uses **PETSc BiCGStab + GAMG algebraic multigrid**
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md-36-  (iteration count ~resolution-independent, O(5–15) V-cycles) for the
--
docs/performance/scaling/scaling_gpu.md-792-
docs/performance/scaling/scaling_gpu.md-793-All 3 configurations stable. impl_cn preserves |u| ~75% better than
docs/performance/scaling/scaling_gpu.md-794-explicit (less numerical dissipation in the barotropic mode);
docs/performance/scaling/scaling_gpu.md-795-geostrophic-adjustment eta amplification is similar across solvers.
docs/performance/scaling/scaling_gpu.md-796-
docs/performance/scaling/scaling_gpu.md-797-PR #319 description updated to reflect iter 21-25 corrections
docs/performance/scaling/scaling_gpu.md-798-(unit-bug walkback, CFL-validated nsub recommendations, final ladder).
docs/performance/scaling/scaling_gpu.md-799-
docs/performance/scaling/scaling_gpu.md:800:### Iter 25 — 2026-05-27 — MPAS CFL stability sweep CONFIRMS iter-22 nsub=10
docs/performance/scaling/scaling_gpu.md-801-
docs/performance/scaling/scaling_gpu.md-802-Mirror iter-24 check on MPAS. I5 fp64 dt=600 s, 200 steps (33 h) with
docs/performance/scaling/scaling_gpu.md-803-eta=0.1 m kick:
docs/performance/scaling/scaling_gpu.md-804-
docs/performance/scaling/scaling_gpu.md-805-| nsub | finite? | |eta|_max  |
docs/performance/scaling/scaling_gpu.md-806-|------|---------|------------|
docs/performance/scaling/scaling_gpu.md-807-|   5  | True    | 0.0376 m   |
docs/performance/scaling/scaling_gpu.md-808-|  10  | True    | 0.0453 m   |
--
docs/performance/scaling/levante_campaign_2026-07-24.md-97-Mc/s/GPU): 2->8 = 2.88x = eff 0.72, 2->16 = 2.80x = eff 0.35 (small-tile
docs/performance/scaling/levante_campaign_2026-07-24.md-98-floor).
docs/performance/scaling/levante_campaign_2026-07-24.md-99-
docs/performance/scaling/levante_campaign_2026-07-24.md-100-OPEN ANOMALY, characterised not explained: per-GPU throughput dips at
docs/performance/scaling/levante_campaign_2026-07-24.md-101-np=4 (247 Mc/s/GPU vs 430 at np2 and 306 at np8), so np4 is barely faster
docs/performance/scaling/levante_campaign_2026-07-24.md-102-than np2 while np8 is 2.5x faster than np4. Evidence gathered:
docs/performance/scaling/levante_campaign_2026-07-24.md-103-- REPRODUCIBLE: two repeats per arm agree within 1 % (19.92/19.87,
docs/performance/scaling/levante_campaign_2026-07-24.md-104-  17.14/17.04, 6.87/6.97).
docs/performance/scaling/levante_campaign_2026-07-24.md:105:- PLACEMENT REFUTED: np4 packed on one node (17.09 ms) == np4 spread over
docs/performance/scaling/levante_campaign_2026-07-24.md-106-  two nodes (17.12 ms), so node crossing is irrelevant.
docs/performance/scaling/levante_campaign_2026-07-24.md:107:- HALO VOLUME REFUTED: ghost-cell census on the padded mesh gives
docs/performance/scaling/levante_campaign_2026-07-24.md-108-  1540/1587/1400/1136 ghost cells per device at np 2/4/8/16 — flat to
docs/performance/scaling/levante_campaign_2026-07-24.md-109-  falling, and under 3 % of owned cells at every count.
docs/performance/scaling/levante_campaign_2026-07-24.md:110:- COLLECTIVE COUNT REFUTED (HLO census, ico L7, CPU virtual devices —
docs/performance/scaling/levante_campaign_2026-07-24.md-111-  device count is a compile-time property so the HLO matches what the GPUs
docs/performance/scaling/levante_campaign_2026-07-24.md:112:  execute): collective-permutes per step are 3 / 9 / 21 at np 2/4/8, i.e.
docs/performance/scaling/levante_campaign_2026-07-24.md-113-  np8 issues 2.3x MORE collectives than np4 and still runs 2.5x faster.
docs/performance/scaling/levante_campaign_2026-07-24.md-114-  Collective COUNT therefore cannot explain the np4 dip (this assumes cost
docs/performance/scaling/levante_campaign_2026-07-24.md-115-rises with count; a per-message-size effect is not excluded). (Fusion count 136/173/240,
docs/performance/scaling/levante_campaign_2026-07-24.md-116-  bitcasts 526/582/694 — the np8 program is finer-grained.)
docs/performance/scaling/levante_campaign_2026-07-24.md:117:- PARTITION METHOD REFUTED (job 26455829): the dip is method-independent —
docs/performance/scaling/levante_campaign_2026-07-24.md-118-  np4/np8 = 17.22/7.00 ms (sfc), 17.20/6.97 (metis), 19.35/6.26
docs/performance/scaling/levante_campaign_2026-07-24.md-119-  (geometric). Every method shows the same 2.5-3.1x jump.
docs/performance/scaling/levante_campaign_2026-07-24.md:120:- XLA CODEGEN ENV KNOBS REFUTED (job 26455948): np4 is 17.12 ms base,
docs/performance/scaling/levante_campaign_2026-07-24.md-121-  17.06 autotune-level-4, 17.10 latency-hiding-off, 17.01
docs/performance/scaling/levante_campaign_2026-07-24.md-122-  command-buffers-off — every arm within 1 %, none recovers np4.
docs/performance/scaling/levante_campaign_2026-07-24.md-123-  (The multi-output-fusion arm errored on an unsupported flag name and is
docs/performance/scaling/levante_campaign_2026-07-24.md-124-  not counted.)
docs/performance/scaling/levante_campaign_2026-07-24.md-125-VERDICT: five hypotheses refuted by measurement (placement, halo volume,
docs/performance/scaling/levante_campaign_2026-07-24.md-126-collective count, partition method, codegen env knobs). The cheap levers known to this campaign are exhausted; the remaining suspect — per-device kernel efficiency
docs/performance/scaling/levante_campaign_2026-07-24.md-127-for this shape — needs a GPU op-level profile (nsys / XLA op profile of
docs/performance/scaling/levante_campaign_2026-07-24.md-128-np4 vs np8), which is a separate instrumented project, not another timing
docs/performance/scaling/levante_campaign_2026-07-24.md-129-run. Per-GPU throughput across the ladder is non-monotone in tile size
docs/performance/scaling/levante_campaign_2026-07-24.md-130-(430 / 249 / 305 / 150 Mc/s/GPU at 327688 / 163844 / 81922 / 40961
docs/performance/scaling/levante_campaign_2026-07-24.md-131-cells/device), which is itself the clue to hand the profiler.
docs/performance/scaling/levante_campaign_2026-07-24.md-132-PRACTICAL GUIDANCE MEANWHILE: run this grid at np>=8, where per-device
docs/performance/scaling/levante_campaign_2026-07-24.md-133-throughput is 304-340 Mc/s/GPU vs 220-248 at np4.
docs/performance/scaling/levante_campaign_2026-07-24.md-134-
docs/performance/scaling/levante_campaign_2026-07-24.md-135-RESOLVED 2026-07-26 (nsys job 26479922 + HLO dump 26480096 + sqlite
docs/performance/scaling/levante_campaign_2026-07-24.md-136-timeline): the dip is an XLA CODEGEN pathology, localized to named
docs/performance/scaling/levante_campaign_2026-07-24.md:137:kernels. CONFIRMED: (1) the dip reproduces under nsys with matched
docs/performance/scaling/levante_campaign_2026-07-24.md-138-protocol (L8, padded-16 mesh: 21.01/17.78/7.17 ms at np2/4/8 vs campaign
docs/performance/scaling/levante_campaign_2026-07-24.md-139-19.90/17.09/6.92 — ~5% profiler overhead); (2) at np4 ONLY, giant
docs/performance/scaling/levante_campaign_2026-07-24.md-140-serialized "loop fusion" kernels appear — loop_add_fusion_1/2 at 3.6 ms
docs/performance/scaling/levante_campaign_2026-07-24.md-141-per launch (vs ~3 us for ordinary elementwise kernels) plus a THIRD
docs/performance/scaling/levante_campaign_2026-07-24.md-142-once-per-step group (the unsuffixed loop_add_fusion: 12 of its 44
docs/performance/scaling/levante_campaign_2026-07-24.md-143-instances are >1 ms at ~3.3 ms, the rest are the ordinary us-scale adds)
docs/performance/scaling/levante_campaign_2026-07-24.md-144-— and the sqlite timeline places all three groups' big instances at the
docs/performance/scaling/levante_campaign_2026-07-24.md-145-17.8 ms step cadence (stddev 78 us: deterministic compute, not comm
--
docs/performance/scaling/levante_campaign_2026-07-24.md-208-device count (jobs 26456334 / 26457693 / 26456337 vs 26452804-06).
docs/performance/scaling/levante_campaign_2026-07-24.md-209-
docs/performance/scaling/levante_campaign_2026-07-24.md-210-So the ocean shows the SAME tile-size dependence the cube does: the 2.01x
docs/performance/scaling/levante_campaign_2026-07-24.md-211-multinode improvement measured at LL576 was partly a floor effect, and at
docs/performance/scaling/levante_campaign_2026-07-24.md-212-a production tile the identical code scales substantially better (0.37 ->
docs/performance/scaling/levante_campaign_2026-07-24.md-213-0.63). Per-device throughput also rises (259 -> 305 Mc/s/GPU at np4).
docs/performance/scaling/levante_campaign_2026-07-24.md-214-Config is byte-identical between the two rows; only the grid changes.
docs/performance/scaling/levante_campaign_2026-07-24.md-215-
docs/performance/scaling/levante_campaign_2026-07-24.md:216:REFUTED EN ROUTE: the np8 leg timed out twice (>90 min still tracing) while
docs/performance/scaling/levante_campaign_2026-07-24.md-217-np4 — a LARGER per-device tile — finished in ~25 min, which looked like a
docs/performance/scaling/levante_campaign_2026-07-24.md-218-compile-time cliff at that device count. It is not: the third attempt ran
docs/performance/scaling/levante_campaign_2026-07-24.md-219-the identical configuration in **99 seconds** with a 21.6 s compile (job
docs/performance/scaling/levante_campaign_2026-07-24.md-220-26457693). The earlier hangs were transient/environmental, not
docs/performance/scaling/levante_campaign_2026-07-24.md-221-reproducible, and no compile-time defect is claimed.
docs/performance/scaling/levante_campaign_2026-07-24.md-222-
docs/performance/scaling/levante_campaign_2026-07-24.md-223-## Weak scaling at production per-device size (job 26453523)
docs/performance/scaling/levante_campaign_2026-07-24.md-224-
--
docs/performance/scaling/levante_campaign_2026-07-24.md-347-| 2 | 42.71 ms | 34.77 ms | 1.228 | 81 % |
docs/performance/scaling/levante_campaign_2026-07-24.md-348-| 4 | 23.53 ms | 16.82 ms | 1.398 | 72 % |
docs/performance/scaling/levante_campaign_2026-07-24.md-349-
docs/performance/scaling/levante_campaign_2026-07-24.md-350-Bound = per-device compute (32.58 / 14.63 ms) + modelled comm (2.21) +
docs/performance/scaling/levante_campaign_2026-07-24.md-351-modelled reduction (2.19). The unmodelled gap is **~5 ms/step and roughly
docs/performance/scaling/levante_campaign_2026-07-24.md-352-FLAT** with device count (7.9 ms at nd2, 6.7 at nd4), which is why the
docs/performance/scaling/levante_campaign_2026-07-24.md-353-ratio worsens as compute shrinks.
docs/performance/scaling/levante_campaign_2026-07-24.md-354-
docs/performance/scaling/levante_campaign_2026-07-24.md:355:WHAT THE GAP IS NOT — the omitted-traffic explanation is REFUTED
docs/performance/scaling/levante_campaign_2026-07-24.md-356-(`scripts/tmp/probe_ocean_halo_bytes.py`, HLO byte census on CPU virtual
docs/performance/scaling/levante_campaign_2026-07-24.md-357-devices). The bench's `comm_scope_note` correctly warns that its census is
docs/performance/scaling/levante_campaign_2026-07-24.md-358-"barotropic implicit-CN PCG scope only … baroclinic 3-D pads NOT counted",
docs/performance/scaling/levante_campaign_2026-07-24.md-359-and the true volume IS much larger: **16.22 MB/step across 110
docs/performance/scaling/levante_campaign_2026-07-24.md:360:collective-permutes vs the censused 2.25 MB — a 7.2x undercount**. But
docs/performance/scaling/levante_campaign_2026-07-24.md-361-completing the census moves the bound by only **0.22 ms**, because the
docs/performance/scaling/levante_campaign_2026-07-24.md-362-comm term is LATENCY-dominated: at 122 messages x 17.82 us the latency part
docs/performance/scaling/levante_campaign_2026-07-24.md-363-is 2.174 ms while even 16 MB at 64.22 GB/s is just 0.253 ms.
docs/performance/scaling/levante_campaign_2026-07-24.md-364-
docs/performance/scaling/levante_campaign_2026-07-24.md-365-So with the byte census completed the unexplained residual is still 5.5 ms
docs/performance/scaling/levante_campaign_2026-07-24.md-366-(nd2) and 4.3 ms (nd4).
docs/performance/scaling/levante_campaign_2026-07-24.md-367-
docs/performance/scaling/levante_campaign_2026-07-24.md:368:SECOND CANDIDATE ALSO REFUTED (`scripts/tmp/probe_sharded_overhead.py`,
docs/performance/scaling/levante_campaign_2026-07-24.md-369-job 26458553): the sharded formulation does NOT do measurably more work.
docs/performance/scaling/levante_campaign_2026-07-24.md-370-Timing the SHARDED step on a 1-device mesh (all the padding, band-edge and
docs/performance/scaling/levante_campaign_2026-07-24.md-371-v-row-reconstruction machinery present, ppermutes self-to-self so no real
docs/performance/scaling/levante_campaign_2026-07-24.md-372-traffic) against the UNSHARDED step at the identical tile:
docs/performance/scaling/levante_campaign_2026-07-24.md-373-
docs/performance/scaling/levante_campaign_2026-07-24.md-374-| tile | unsharded | sharded on 1 device | overhead |
docs/performance/scaling/levante_campaign_2026-07-24.md-375-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-376-| 288x1152x20 | 33.13 ms | 32.79 ms | **-0.34 ms (-1.0 %)** |
--
docs/performance/scaling/levante_campaign_2026-07-24.md-380-reference and extra sharded work is not the gap.
docs/performance/scaling/levante_campaign_2026-07-24.md-381-
docs/performance/scaling/levante_campaign_2026-07-24.md-382-WHERE THAT LEAVES IT (quantified, one candidate standing): the residual
docs/performance/scaling/levante_campaign_2026-07-24.md-383-divided by the message count is **83 us/message at nd2 and 73 us at nd4**,
docs/performance/scaling/levante_campaign_2026-07-24.md-384-versus **17.8 us** for the same collective measured in isolation — an in-
docs/performance/scaling/levante_campaign_2026-07-24.md-385-context cost 4-5x the best case. That is consistent with EXPOSED,
docs/performance/scaling/levante_campaign_2026-07-24.md-386-un-overlapped communication rather than raw wire time.
docs/performance/scaling/levante_campaign_2026-07-24.md-387-
docs/performance/scaling/levante_campaign_2026-07-24.md:388:THIRD CANDIDATE REFUTED, AND IT IDENTIFIES THE MECHANISM (job 26458930).
docs/performance/scaling/levante_campaign_2026-07-24.md-389-If the residual were communication the scheduler is currently hiding work
docs/performance/scaling/levante_campaign_2026-07-24.md-390-behind, DISABLING XLA's latency-hiding scheduler would hurt. It does not:
docs/performance/scaling/levante_campaign_2026-07-24.md-391-
docs/performance/scaling/levante_campaign_2026-07-24.md-392-| arm | nd2 | nd4 | vs default |
docs/performance/scaling/levante_campaign_2026-07-24.md-393-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-394-| default (LHS on) | 42.69 | 23.53 ms | — |
docs/performance/scaling/levante_campaign_2026-07-24.md-395-| `latency_hiding_scheduler=false` | 42.44 | 23.49 | **+0.6 % / +0.2 %** |
docs/performance/scaling/levante_campaign_2026-07-24.md:396:| `enable_pipelined_p2p=true` | 42.76 | 23.54 | -0.2 % / -0.1 % |
docs/performance/scaling/levante_campaign_2026-07-24.md-397-| CP combining @32 MiB | 42.55 | 23.59 | +0.3 % / -0.3 % |
docs/performance/scaling/levante_campaign_2026-07-24.md-398-
docs/performance/scaling/levante_campaign_2026-07-24.md-399-Turning overlap OFF is free (marginally faster), and no scheduling flag
docs/performance/scaling/levante_campaign_2026-07-24.md-400-moves the step. The scheduler has nothing to hide the comm behind.
docs/performance/scaling/levante_campaign_2026-07-24.md-401-
docs/performance/scaling/levante_campaign_2026-07-24.md-402-**MECHANISM (the three tested alternatives are not dominant): the
docs/performance/scaling/levante_campaign_2026-07-24.md-403-residual is best explained by EXPOSED, DEPENDENCY-SERIALIZED
docs/performance/scaling/levante_campaign_2026-07-24.md-404-SYNCHRONISATION.** These are eliminations of the TESTED implementations,
--
docs/performance/scaling/levante_campaign_2026-07-24.md-451-
docs/performance/scaling/levante_campaign_2026-07-24.md-452-WHAT THIS ESTABLISHES (codex objection (c), answered): the fit is
docs/performance/scaling/levante_campaign_2026-07-24.md-453-essentially exact and the intercept is tightly determined, so the cost is
docs/performance/scaling/levante_campaign_2026-07-24.md-454-genuinely PER-ITERATION, not a constant misattributed to iterations. But
docs/performance/scaling/levante_campaign_2026-07-24.md-455-the nd4 intercept EXCEEDS the measured 144-row compute term (14.63 ms) by
docs/performance/scaling/levante_campaign_2026-07-24.md-456-**1.48 ms**, so a fixed non-PCG overhead does exist and the 60 iterations
docs/performance/scaling/levante_campaign_2026-07-24.md-457-do NOT explain the entire residual.
docs/performance/scaling/levante_campaign_2026-07-24.md-458-
docs/performance/scaling/levante_campaign_2026-07-24.md:459:CODEX OBJECTION (a) TESTED AND REFUTED (job 26459817). Rather than divide
docs/performance/scaling/levante_campaign_2026-07-24.md-460-the nd1 slope by 4, measure the per-iteration slope directly at each tile
docs/performance/scaling/levante_campaign_2026-07-24.md-461-on ONE device:
docs/performance/scaling/levante_campaign_2026-07-24.md-462-
docs/performance/scaling/levante_campaign_2026-07-24.md-463-| tile | measured us/iter |
docs/performance/scaling/levante_campaign_2026-07-24.md-464-|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-465-| 576 rows | 110.0 |
docs/performance/scaling/levante_campaign_2026-07-24.md-466-| 288 rows | 65.2 |
docs/performance/scaling/levante_campaign_2026-07-24.md-467-| 144 rows | **28.1** |
--
docs/performance/scaling/levante_campaign_2026-07-24.md-637-stability gate — that needs the filter analysis under stale halos and a
docs/performance/scaling/levante_campaign_2026-07-24.md-638-science sign-off — but "no observed failure" has become "marginally better
docs/performance/scaling/levante_campaign_2026-07-24.md-639-conservation with a measured variance estimate".
docs/performance/scaling/levante_campaign_2026-07-24.md-640-
docs/performance/scaling/levante_campaign_2026-07-24.md-641-Per-step time is flat 100 -> 600 steps (12.81 -> 12.74 ms). Both arms' heat
docs/performance/scaling/levante_campaign_2026-07-24.md-642-drift grows ~linearly and is similar between them, consistent with the
docs/performance/scaling/levante_campaign_2026-07-24.md-643-shared baroclinic/tracer path dominating it.
docs/performance/scaling/levante_campaign_2026-07-24.md-644-
docs/performance/scaling/levante_campaign_2026-07-24.md:645:MECHANISM CONFIRMED FROM A THIRD ANGLE. The wide-halo arm records its own
docs/performance/scaling/levante_campaign_2026-07-24.md-646-message census: **120 standard barotropic messages/step -> 4** (n_loop=30
docs/performance/scaling/levante_campaign_2026-07-24.md-647-substeps, stencil reach 3, one fixed wide exchange per chunk). A 30x cut in
docs/performance/scaling/levante_campaign_2026-07-24.md-648-barotropic exchanges is precisely why it wins where sync dominates, and it
docs/performance/scaling/levante_campaign_2026-07-24.md-649-is the SAME quantity the single_reduce analysis isolated as the half it
docs/performance/scaling/levante_campaign_2026-07-24.md-650-could not touch (44.1 us/iter of matvec halo). Three independent
docs/performance/scaling/levante_campaign_2026-07-24.md-651-measurements — the iteration sweep, the single_reduce decomposition and
docs/performance/scaling/levante_campaign_2026-07-24.md-652-this census — now agree on what the cost is.
docs/performance/scaling/levante_campaign_2026-07-24.md-653-
--
docs/performance/scaling/levante_campaign_2026-07-24.md-979-Unblocking #1370 and sourcing a subdiv-9+ mesh are therefore worth more
docs/performance/scaling/levante_campaign_2026-07-24.md-980-than any further tuning: both lanes have headroom that is currently
docs/performance/scaling/levante_campaign_2026-07-24.md-981-unreachable.
docs/performance/scaling/levante_campaign_2026-07-24.md-982-
docs/performance/scaling/levante_campaign_2026-07-24.md-983-## Cube optimisation: bounded BEFORE implementing — and the bound killed the plan
docs/performance/scaling/levante_campaign_2026-07-24.md-984-
docs/performance/scaling/levante_campaign_2026-07-24.md-985-Directive was to push the cube toward its limit. Codex round-16 defined
docs/performance/scaling/levante_campaign_2026-07-24.md-986-the strategy first (per the standing pre-implementation rule) and its
docs/performance/scaling/levante_campaign_2026-07-24.md:987:cheapest-bound step then REFUTED the intervention I was about to build.
docs/performance/scaling/levante_campaign_2026-07-24.md-988-
docs/performance/scaling/levante_campaign_2026-07-24.md-989-**Codex corrections to my reading:**
docs/performance/scaling/levante_campaign_2026-07-24.md-990-* My "88 SendRecv / 4 rounds = 22 exchanges" was wrong. The kt=2 tile pad
docs/performance/scaling/levante_campaign_2026-07-24.md-991-  emits **16 phases per logical scalar pad** (4 edges + 4 guards + 4
docs/performance/scaling/levante_campaign_2026-07-24.md-992-  diagonals + 4 corner slivers) for serial-exact offset/corner handling
docs/performance/scaling/levante_campaign_2026-07-24.md-993-  (`cubesphere_exchange.py:1276`); the RK body pads dp/B/zeta/invT/lnps/T
docs/performance/scaling/levante_campaign_2026-07-24.md-994-  separately, the vector pad calls the scalar pad twice, x3 RK stages.
docs/performance/scaling/levante_campaign_2026-07-24.md-995-* Vertical batching is ALREADY done (4-D pads carry all L60 levels), so
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1470-   were always sound (cube C96/L40: 0/7/14 CP/step at nd 1/2/3 + 1
docs/performance/scaling/levante_campaign_2026-07-24.md-1471-   all-reduce, job 26447827; 24-dev tiled closed loop 384 CP + 1 AR,
docs/performance/scaling/levante_campaign_2026-07-24.md-1472-   job 26450938).
docs/performance/scaling/levante_campaign_2026-07-24.md-1473-
docs/performance/scaling/levante_campaign_2026-07-24.md-1474-## Closed levers (nulls with receipts — do not re-run)
docs/performance/scaling/levante_campaign_2026-07-24.md-1475-
docs/performance/scaling/levante_campaign_2026-07-24.md-1476-NCCL_PROTO forcing (default already optimal; LL128 −8–12%), fused-halo on
docs/performance/scaling/levante_campaign_2026-07-24.md-1477-the implicit arm AND on the wide arm (pad aggregation is not the residual),
docs/performance/scaling/levante_campaign_2026-07-24.md:1478:`xla_gpu_collective_permute_combine_threshold_bytes` alone,
docs/performance/scaling/levante_campaign_2026-07-24.md:1479:`--xla_gpu_enable_pipelined_p2p` alone, `LEGOESM_BAROCLINIC_F32` (~+0.5%, within run-to-run spread; job 26451282).
docs/performance/scaling/levante_campaign_2026-07-24.md-1480-PGLE arm invalid as measured here (the 33-step window catches its
docs/performance/scaling/levante_campaign_2026-07-24.md-1481-profile+recompile). The +8.5% figure is from the DERECHO lane-T campaign
docs/performance/scaling/levante_campaign_2026-07-24.md-1482-(see the SOTA review), not reproduced on Levante — rerun long-window if
docs/performance/scaling/levante_campaign_2026-07-24.md-1483-revisited.
docs/performance/scaling/levante_campaign_2026-07-24.md-1484-
docs/performance/scaling/levante_campaign_2026-07-24.md-1485-## Still open (ranked) — refreshed at campaign close
docs/performance/scaling/levante_campaign_2026-07-24.md-1486-
docs/performance/scaling/levante_campaign_2026-07-24.md-1487-1. **OMIP config sign-off for explicit_substep+wide** (formerly "wide-halo
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1528-   negative on this codebase — the "~2x on ~40%-land grids" projection is
docs/performance/scaling/levante_campaign_2026-07-24.md-1529-   refuted twice over (gather penalty eats the compaction saving at 0.71
docs/performance/scaling/levante_campaign_2026-07-24.md-1530-   wet; wet-balanced bands worsen dense-compute balance). The item is
docs/performance/scaling/levante_campaign_2026-07-24.md-1531-   CLOSED as not-worth-building, with receipts. (Also moot for the SPMD
docs/performance/scaling/levante_campaign_2026-07-24.md-1532-   lane: jax equal-shard sharding would need padding to the max band,
docs/performance/scaling/levante_campaign_2026-07-24.md-1533-   returning exactly the imbalance removed.)
docs/performance/scaling/levante_campaign_2026-07-24.md-1534-
docs/performance/scaling/levante_campaign_2026-07-24.md-1535-   *Expensive half — gather/scatter compaction: MEASURED, and the audit's
docs/performance/scaling/levante_campaign_2026-07-24.md:1536:   "~2x" is REFUTED* (`bench_gather_vs_slice_stencil.py`, job 26479884,
docs/performance/scaling/levante_campaign_2026-07-24.md-1537-   A100 f32, correctness self-checked). Per-cell gather penalty for a
docs/performance/scaling/levante_campaign_2026-07-24.md-1538-   5-point Laplacian vs the dense sliced version: **1.40-1.77x**, so
docs/performance/scaling/levante_campaign_2026-07-24.md-1539-   compaction wins only when wet_fraction < 0.56-0.72 (size-dependent).
docs/performance/scaling/levante_campaign_2026-07-24.md-1540-   At the REAL global-ocean wet fraction (~0.71), packed-gather is a net
docs/performance/scaling/levante_campaign_2026-07-24.md-1541-   LOSS on the full LL576 grid (ratio 1.16) and a wash at the nd4 tile
docs/performance/scaling/levante_campaign_2026-07-24.md-1542-   (1.03). SCOPE (codex round-10): the two numbers are END-MEMBER ESTIMATES, not
docs/performance/scaling/levante_campaign_2026-07-24.md-1543-   a bound — 0.86x is the pure-stencil member (measured), 1.41x the
docs/performance/scaling/levante_campaign_2026-07-24.md-1544-   pure-column ideal; a real step also pays packing/scattering at the
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1675-
docs/performance/scaling/levante_campaign_2026-07-24.md-1676-1. **Ensemble receipt** (`scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch`)
docs/performance/scaling/levante_campaign_2026-07-24.md-1677-   — codex lever #1: 4 concurrent 32-GPU s9 replicas on disjoint 8-node
docs/performance/scaling/levante_campaign_2026-07-24.md-1678-   sets vs SAME-JOB solo controls bracketing phase B (solo before AND
docs/performance/scaling/levante_campaign_2026-07-24.md-1679-   after — BRACKETED, not fully counterbalanced; a penalty's attribution
docs/performance/scaling/levante_campaign_2026-07-24.md-1680-   to fabric vs placement/drift needs the per-step nodelist table +
docs/performance/scaling/levante_campaign_2026-07-24.md-1681-   follow-up). steps=5000 so the stepping window
docs/performance/scaling/levante_campaign_2026-07-24.md-1682-   (~60 s) dwarfs launch skew; per-arm `SLURM_STEP_NODELIST` +
docs/performance/scaling/levante_campaign_2026-07-24.md:1683:   wall-clock brackets logged as overlap evidence. CONFIRM bar:
docs/performance/scaling/levante_campaign_2026-07-24.md-1684-   max(replica) <= 1.10x mean(solo) => guaranteed aggregate >= 3.64x the
docs/performance/scaling/levante_campaign_2026-07-24.md-1685-   32-GPU solo rate (>= 19.9 GC/s if solo reproduces 5.47) = ~3.3x the
docs/performance/scaling/levante_campaign_2026-07-24.md:1686:   observed 128-GPU single-trajectory rate. REFUTE: replica slowdown
docs/performance/scaling/levante_campaign_2026-07-24.md-1687-   >10 % = a CO-EXECUTION penalty, quantified per replica — its
docs/performance/scaling/levante_campaign_2026-07-24.md-1688-   attribution (fabric contention vs placement/topology vs drift) is a
docs/performance/scaling/levante_campaign_2026-07-24.md-1689-   follow-up, not a conclusion of this job.
docs/performance/scaling/levante_campaign_2026-07-24.md-1690-2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
docs/performance/scaling/levante_campaign_2026-07-24.md-1691-   (81.9k/41.0k/20.5k cells/GPU) on the SAME lloyd=0 family, same sfc +
docs/performance/scaling/levante_campaign_2026-07-24.md-1692-   `--reorder-for 128`, same steps/warmup as the s9 ladder. Weak pairs
docs/performance/scaling/levante_campaign_2026-07-24.md-1693-   recomputed only from these.
docs/performance/scaling/levante_campaign_2026-07-24.md-1694-
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1769-81,936/81,924, 40,968/40,962, 20,484/20,481), SAME lloyd-0 family:
docs/performance/scaling/levante_campaign_2026-07-24.md-1770-
docs/performance/scaling/levante_campaign_2026-07-24.md-1771-| cells/GPU | s8 rung | s9 rung | ratio | weak eff |
docs/performance/scaling/levante_campaign_2026-07-24.md-1772-|---|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-1773-| 81.9k | np8 6.58 | np32 12.47 | 1.895 | **0.53** |
docs/performance/scaling/levante_campaign_2026-07-24.md-1774-| 41.0k | np16 6.43 | np64 9.60 | 1.493 | 0.67 |
docs/performance/scaling/levante_campaign_2026-07-24.md-1775-| 20.5k | np32 7.29 | np128 11.48 | 1.575 | 0.64 |
docs/performance/scaling/levante_campaign_2026-07-24.md-1776-
docs/performance/scaling/levante_campaign_2026-07-24.md:1777:* The falsifiability block's CONFIRM branch fires: ratios stay well
docs/performance/scaling/levante_campaign_2026-07-24.md-1778-  above 1 with the known Lloyd-family mismatch REMOVED. (This does not
docs/performance/scaling/levante_campaign_2026-07-24.md-1779-  prove the old confound "only" biased the size — these are
docs/performance/scaling/levante_campaign_2026-07-24.md-1780-  unreplicated single runs from separate allocations, one comparator
docs/performance/scaling/levante_campaign_2026-07-24.md-1781-  without row-level provenance; the confounded draft read
docs/performance/scaling/levante_campaign_2026-07-24.md-1782-  1.80/1.35/1.41 vs 1.90/1.49/1.57 here, and the production-mesh s8
docs/performance/scaling/levante_campaign_2026-07-24.md-1783-  np8 was 6.92 vs lloyd-0 6.58, -4.9 %, so the mesh family does shift
docs/performance/scaling/levante_campaign_2026-07-24.md-1784-  absolutes.)
docs/performance/scaling/levante_campaign_2026-07-24.md-1785-* Restated: at NEAR-matched per-GPU tile, ~quadrupling devices+problem
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1794-
docs/performance/scaling/levante_campaign_2026-07-24.md-1795-## Distance-to-modeled-limit: atm lat-lon GPU (2026-08-02, "near theoretical limit" directive)
docs/performance/scaling/levante_campaign_2026-07-24.md-1796-
docs/performance/scaling/levante_campaign_2026-07-24.md-1797-Closed the bench's own honest-null bound gap (audit item 4) for the
docs/performance/scaling/levante_campaign_2026-07-24.md-1798-lat-lon lane, using only repo instruments:
docs/performance/scaling/levante_campaign_2026-07-24.md-1799-
docs/performance/scaling/levante_campaign_2026-07-24.md-1800-* **Halo census** (new probe `scripts/tmp/_probe_latlon_halo_census.py`,
docs/performance/scaling/levante_campaign_2026-07-24.md-1801-  virtual-CPU forced-host-platform lowering of the REAL
docs/performance/scaling/levante_campaign_2026-07-24.md:1802:  `make_sharded_atm_latlon_step`): **41 collective-permutes + 1
docs/performance/scaling/levante_campaign_2026-07-24.md-1803-  all-reduce per step**, nd-INDEPENDENT (identical at nd=8 and nd=16 —
docs/performance/scaling/levante_campaign_2026-07-24.md-1804-  the 1-D band structure check). Exact CP payload from compiled-HLO
docs/performance/scaling/levante_campaign_2026-07-24.md-1805-  result shapes: 4,635,408 B/dev/step at n_lon=1024 L26 f32 = 1.06x the
docs/performance/scaling/levante_campaign_2026-07-24.md-1806-  single-row slab model; linear in n_lon (checked 1024 vs 2048, 0.07 %
docs/performance/scaling/levante_campaign_2026-07-24.md-1807-  residual) -> **18.5 MB/dev/step at n_lon=4096 f32**. CAVEAT: CPU
docs/performance/scaling/levante_campaign_2026-07-24.md-1808-  lowering; GPU-side collective combining could change the executed
docs/performance/scaling/levante_campaign_2026-07-24.md-1809-  count (metadata.py:214) — the bound is a MODEL.
docs/performance/scaling/levante_campaign_2026-07-24.md:1810:* **Same-tile nd=1 compute baselines** (job 26630370, roofline recipe):
docs/performance/scaling/levante_campaign_2026-07-24.md-1811-  16x4096 f32 1.659 ms, 32x4096 f32 2.837, 16x4096 f64 2.973.
docs/performance/scaling/levante_campaign_2026-07-24.md-1812-  Approximation, recorded: nd=1 includes pole tiles -> compute term
docs/performance/scaling/levante_campaign_2026-07-24.md-1813-  biased HIGH -> bound conservative.
docs/performance/scaling/levante_campaign_2026-07-24.md:1814:* **Calibrated bound** (`metadata.calibrated_bound`, measured fabric
docs/performance/scaling/levante_campaign_2026-07-24.md-1815-  constants: IB 26.3 us / 23.5 GB/s, NVLink 17.8 / 64.2):
docs/performance/scaling/levante_campaign_2026-07-24.md-1816-
docs/performance/scaling/levante_campaign_2026-07-24.md-1817-| row | measured | t_bound (IB) | measured/bound |
docs/performance/scaling/levante_campaign_2026-07-24.md-1818-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-1819-| LL2048@64 f32 | 6.732 | 2.863 | **2.35** |
docs/performance/scaling/levante_campaign_2026-07-24.md-1820-| LL2048@128 f32 | 5.577 | 1.848 | **3.02** |
docs/performance/scaling/levante_campaign_2026-07-24.md-1821-| LL2048@128 f64 | 9.602 | 2.999 | **3.20** |
docs/performance/scaling/levante_campaign_2026-07-24.md-1822-
docs/performance/scaling/levante_campaign_2026-07-24.md-1823-* **The lat-lon GPU panel sits ~2.4-3.2x ABOVE its modeled limit** — the
docs/performance/scaling/levante_campaign_2026-07-24.md-1824-  eff-0.60 strong leg is NOT close to the fabric+compute floor. Leading
docs/performance/scaling/levante_campaign_2026-07-24.md-1825-  PLAUSIBLE mechanism (uninstrumented): effective per-CP cost
docs/performance/scaling/levante_campaign_2026-07-24.md-1826-  (launch + schedule + stream sync) well above the raw 26 us fabric
docs/performance/scaling/levante_campaign_2026-07-24.md-1827-  latency across 41 dependency-chained exchanges — the arXiv:2607.16100
docs/performance/scaling/levante_campaign_2026-07-24.md-1828-  small-collective regime. The model itself notes the serialized-latency
docs/performance/scaling/levante_campaign_2026-07-24.md-1829-  vs overlap biases pull opposite ways; treat measured/bound as a
docs/performance/scaling/levante_campaign_2026-07-24.md-1830-  consistency diagnostic, not proven headroom.
docs/performance/scaling/levante_campaign_2026-07-24.md:1831:* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
docs/performance/scaling/levante_campaign_2026-07-24.md:1832:  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
docs/performance/scaling/levante_campaign_2026-07-24.md:1833:  same-job control, falsifiability block in the script. The ocean-lane
docs/performance/scaling/levante_campaign_2026-07-24.md-1834-  null for these flags was reduction-dominated — first atm test.
--
docs/performance/scaling/literature_parallelization_2026-06.md-55-   halos in ordered directional ppermutes (diagonals relay through faces —
docs/performance/scaling/literature_parallelization_2026-06.md-56-   no corner messages) and *skips* halo regions a stencil doesn't read
docs/performance/scaling/literature_parallelization_2026-06.md-57-   (3-0). We already relay corners; the per-operator halo-region *skipping*
docs/performance/scaling/literature_parallelization_2026-06.md-58-   is a census-driven trimming lever for the tiled stage.
docs/performance/scaling/literature_parallelization_2026-06.md-59-7. **GSPMD-auto can silently insert a pathological AllGather consuming 80%
docs/performance/scaling/literature_parallelization_2026-06.md-60-   of runtime; the prescribed detection is the device-profile timeline**
docs/performance/scaling/literature_parallelization_2026-06.md-61-   (jax-ml scaling book — 3-0). Independently confirms our np24
docs/performance/scaling/literature_parallelization_2026-06.md-62-   GSPMD-auto 39× story + HLO-census discipline. (The JEP claim that
docs/performance/scaling/literature_parallelization_2026-06.md:63:   shard_map is "only a surgical escape hatch" was REFUTED 1-2; the
docs/performance/scaling/literature_parallelization_2026-06.md-64-   scaling book positions explicit collectives as first-class for
docs/performance/scaling/literature_parallelization_2026-06.md-65-   comm-critical code.)
docs/performance/scaling/literature_parallelization_2026-06.md-66-8. **Manual comm/compute overlap in shard_map works:** stepwise
docs/performance/scaling/literature_parallelization_2026-06.md-67-   ppermute+partial-compute "collective matmul" removed ~77% of the
docs/performance/scaling/literature_parallelization_2026-06.md-68-   communication overhead vs a blocking AllGather (244 µs vs 311 µs, 224 µs
docs/performance/scaling/literature_parallelization_2026-06.md-69-   unsharded baseline) (3-0). The lighter sibling of our parked deep-halo
docs/performance/scaling/literature_parallelization_2026-06.md-70-   idea — applicable to halo+interior-stencil overlap in the tiled stage.
docs/performance/scaling/literature_parallelization_2026-06.md:71:   (The JEP's own transformer overlap example claim was REFUTED 0-3 — cite
docs/performance/scaling/literature_parallelization_2026-06.md-72-   the scaling book, not the JEP, for this pattern.)
docs/performance/scaling/literature_parallelization_2026-06.md-73-9. **CPU same-node wall is universal, not a legoESM defect.** Veros/JAX on
docs/performance/scaling/literature_parallelization_2026-06.md-74-   8×32-core CPU nodes: within 1.4× of Fortran+MPI, the gap attributed to
docs/performance/scaling/literature_parallelization_2026-06.md-75-   DRAM-bandwidth-bound execution + XLA's incomplete fusion; at high counts
docs/performance/scaling/literature_parallelization_2026-06.md-76-   communication + barotropic dominate (3-0). Keeps our 1-proc/node +
docs/performance/scaling/literature_parallelization_2026-06.md-77-   ranks/node≤8 policies evidence-backed.
docs/performance/scaling/literature_parallelization_2026-06.md-78-10. **GPU saturation rule: ≳10⁶ grid elements per device** before weak
docs/performance/scaling/literature_parallelization_2026-06.md-79-    scaling is meaningful; Veros 16×A100 NVLink strong scaling = 3.4×,
--
docs/performance/scaling/cube_production_tiling_design.md-132-  deep-h pre-pad). h for Bernoulli is taken from the deep-window interior
docs/performance/scaling/cube_production_tiling_design.md-133-  (`hw[:, 3:-3, 3:-3]`). Bit-identity of ALL THREE tendencies vs global at kt=2
docs/performance/scaling/cube_production_tiling_design.md-134-  (np24) + kt=3 (np54), rel<1e-10; codex-clean (8/8 composition vectors). Gate:
docs/performance/scaling/cube_production_tiling_design.md-135-  `test_tiled_fv3_sw_full.py` (job 8482569: TILED_FULL_GATE_OK). The production
docs/performance/scaling/cube_production_tiling_design.md-136-  cube SW dycore tendency now runs on np=6*kt^2 devices.
docs/performance/scaling/cube_production_tiling_design.md-137-- **MULTI-NODE VALIDATION** — `scripts/validate/validate_tiled_fv3_sw_multinode.py`
docs/performance/scaling/cube_production_tiling_design.md-138-  + `scripts/cluster/scaling_ginsburg/tiled_fv3_sw_multinode.sbatch`: runs the
docs/performance/scaling/cube_production_tiling_design.md-139-  full stage under REAL multi-controller `jax.distributed` across 2 nodes (np24,
docs/performance/scaling/cube_production_tiling_design.md:140:  in-stage ppermutes -> cross-NODE collective-permute); each process
docs/performance/scaling/cube_production_tiling_design.md-141-  self-validates its local tile vs serial (rel<1e-9). Correctness, not a bench.
docs/performance/scaling/cube_production_tiling_design.md-142-- **PRODUCTION ASSEMBLY WIRED (2026-07-09)** — the blocked persistent step
docs/performance/scaling/cube_production_tiling_design.md-143-  (`make_tiled_fv3_hydrostatic_step_blocked_2d`, input layout == output
docs/performance/scaling/cube_production_tiling_design.md-144-  layout, in-stage telescoping mass fixer, optional moist column physics)
docs/performance/scaling/cube_production_tiling_design.md-145-  + `make_tiled_cc_loop` (adapter enter/step/exit_) + the
docs/performance/scaling/cube_production_tiling_design.md-146-  `run_cpu_mpi_scaling --cs-spmd` 6·kt² dispatch and
docs/performance/scaling/cube_production_tiling_design.md-147-  `bench_cube_tiled_step_scaling --closed-loop` lane.  Bitwise-identical to
docs/performance/scaling/cube_production_tiling_design.md-148-  the gated step stages (dry + moist gates in
--
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py-376-# exactly ``max_iter`` iterations).  Standard preconditioned CG needs two
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py-377-# sequentially-dependent inner products per iteration — ``p·Ap`` (for α)
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py-378-# and ``r·z`` (for β, which needs the updated ``r`` that depends on α) —
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py-379-# so the loop issues TWO batched ``allreduce(SUM)`` per iteration.  The
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py-380-# residual-monitor ``r·r`` is folded into the SECOND reduction so the
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py-381-# diagnostic costs no extra message.  Two reductions/iter is still a
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py-382-# FIXED count independent of global resolution (the weak-scaling
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py-383-# property) — vastly fewer than ``explicit_substep``'s O(n_substeps ∝
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:384:# resolution) reductions.  A one-reduction pipelined CG
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py-385-# (Chronopoulos-Gear) is possible but changes the rounding and would
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py-386-# break the tight single-rank-vs-stock-CG equivalence pin, so it is not
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py-387-# used.  Every rank runs the identical collective schedule => no
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py-388-# deadlock, JIT-static trace.
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py-389-#
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py-390-# Differentiability — why NOT ``jax.lax.custom_linear_solve``:
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py-391-# the original design wrapped the solve in ``custom_linear_solve`` (for
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py-392-# the implicit-function adjoint, avoiding storing the M primal
--
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-642-    field_yxz: jax.Array, u_at_field: jax.Array, dx: float,
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-643-) -> jax.Array:
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-644-    """2nd-order CENTERED flux-form advection ``-u df/dx`` = gSAM `advect2_mom`.
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-645-
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-646-    ADV-SPLIT #86 (codex iter-68): a 2nd-order CENTERED, NON-diffusive momentum
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-647-    scheme matching SAM's face reconstruction (``advect2_mom_xy.f90:27``:
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-648-    ``flux=0.25·(c_face)·(φ_face)`` — no upwind bias, no flux limiter). The TVD
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-649-    van_leer default adds limiter diffusion that SUPPRESSES convective updraft
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py:650:    cores / w-variance tails (CONFIRMED: identical-IC max|w| caps ~2.6 m/s vs
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-651-    low-diffusion ~5-15; centered restores ~5 in a gray burst). For SAM-faithful
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-652-    convective EXTREMES the momentum legs (u/v/w) use this scheme; the SCALAR legs
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-653-    keep van_leer (monotone ≈ MPDATA, positivity). NOTE (codex iter-68 [S1]): this
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-654-    uses the ADVECTIVE form ``−d(uf)/dx + f·du/dx`` (same identity as van_leer/
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-655-    weno5 — preserves a constant f under divergent flow), which is NOT proven
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-656-    discretely KE-conserving like SAM's pure-flux ``advect2_mom`` — call it
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-657-    "centered non-diffusive matching SAM's face reconstruction", not "energy-
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py-658-    conserving". DISPERSIVE (2Δ modes) — relies on the ∇⁴ hyperdiff + Smagorinsky
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py-541-            _dgr_dy = _dgr_dy_flat.reshape(
packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py-542-                _dgr_dy_flat.shape[0], _dgr_dy_flat.shape[1],
packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py-543-                _dgr_dy_flat.shape[2], nlev_c, 2)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py-544-            corr_dx = (-_dgr_dx[..., 1] + z_ref_corner * _dgr_dx[..., 0]).astype(T.dtype)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py-545-            corr_dy = (-_dgr_dy[..., 1] + z_ref_corner * _dgr_dy[..., 0]).astype(T.dtype)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py-546-            dp_dx = dp_dx + corr_dx
packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py-547-            dp_dy_perp = dp_dy_perp + corr_dy
packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py-548-
packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py:549:    # --- 10c. PGF-ZERO falsification diagnostic (config.pgf_scheme == "zero") ---
packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py-550-    # Remove the horizontal pressure force ENTIRELY.  A rest state then has NO
packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py-551-    # horizontal force at all, so if the cube cold-start stays stable with
packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py-552-    # pgf_scheme="zero" but blows with adcroft/smc03, the partial-cell PGF
packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py-553-    # residual is the SOLE cause (vs any barotropic / advective / metric term).
packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py-554-    # Works for both partial and z* (zeroes the base AL gradient + any
packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py-555-    # correction).  Diagnostic only — never a faithful run.
packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py-556-    if pgf_scheme == "zero":
packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py-557-        dp_dx = jnp.zeros_like(dp_dx)
--
packages/coupler/legoesm/driver/model_driver.py-9342-        solar_weights = ctx["solar_weights"]
packages/coupler/legoesm/driver/model_driver.py-9343-        o3_vmr, aerosol_od = ctx["o3_vmr"], ctx["aerosol_od"]
packages/coupler/legoesm/driver/model_driver.py-9344-
packages/coupler/legoesm/driver/model_driver.py-9345-        seg_len = max(1, int(86400.0 / DT))
packages/coupler/legoesm/driver/model_driver.py-9346-        current_step = start_step
packages/coupler/legoesm/driver/model_driver.py-9347-        status = "COMPLETED"
packages/coupler/legoesm/driver/model_driver.py-9348-        seg_idx = -1
packages/coupler/legoesm/driver/model_driver.py-9349-        # #921: prime every NCCL clique this step uses (the halo
packages/coupler/legoesm/driver/model_driver.py:9350:        # collective-permutes + the target-mass / moisture-fixer psums) in a
packages/coupler/legoesm/driver/model_driver.py-9351-        # fixed, rank-independent order before the first real step, so
packages/coupler/legoesm/driver/model_driver.py-9352-        # multi-process (route-B) comm-init cannot deadlock.  No-op
packages/coupler/legoesm/driver/model_driver.py-9353-        # single-process (CPU-virtual / single-GPU) — those lanes are unchanged.
packages/coupler/legoesm/driver/model_driver.py-9354-        from legoesm.parallel.tiled_production_cdgrid import (
packages/coupler/legoesm/driver/model_driver.py-9355-            warmup_tiled_cube_comms,
packages/coupler/legoesm/driver/model_driver.py-9356-        )
packages/coupler/legoesm/driver/model_driver.py-9357-        warmup_tiled_cube_comms(mesh, kt)
packages/coupler/legoesm/driver/model_driver.py-9358-        t0 = time.time()
--
packages/coupler/legoesm/driver/model_driver.py-9604-                nm: Field(data=self.tracers[nm], name=nm,
packages/coupler/legoesm/driver/model_driver.py-9605-                          dims=self.state.T.dims, units="kg/kg")
packages/coupler/legoesm/driver/model_driver.py-9606-                for nm in ("q_v", "q_c", "q_r")
packages/coupler/legoesm/driver/model_driver.py-9607-            })
packages/coupler/legoesm/driver/model_driver.py-9608-        template = self.state
packages/coupler/legoesm/driver/model_driver.py-9609-        blocked = enter(self.state)
packages/coupler/legoesm/driver/model_driver.py-9610-
packages/coupler/legoesm/driver/model_driver.py-9611-        # #921: prime every NCCL clique the blocked step uses (the halo
packages/coupler/legoesm/driver/model_driver.py:9612:        # collective-permutes + the in-stage mass-fixer psum) in a fixed,
packages/coupler/legoesm/driver/model_driver.py-9613-        # rank-independent order before the first real step, so multi-process
packages/coupler/legoesm/driver/model_driver.py-9614-        # (route-B one-process-per-GPU) comm-init cannot deadlock.  No-op
packages/coupler/legoesm/driver/model_driver.py-9615-        # single-process (CPU-virtual / single-GPU) — those lanes are unchanged.
packages/coupler/legoesm/driver/model_driver.py-9616-        from legoesm.parallel.tiled_production_cdgrid import (
packages/coupler/legoesm/driver/model_driver.py-9617-            warmup_tiled_cube_comms,
packages/coupler/legoesm/driver/model_driver.py-9618-        )
packages/coupler/legoesm/driver/model_driver.py-9619-        warmup_tiled_cube_comms(mesh, kt)
packages/coupler/legoesm/driver/model_driver.py-9620-
--
scripts/tmp/_probe_latlon_halo_census.py-48-mesh = jax.sharding.Mesh(np.array(devs), axis_names=("lat",))
scripts/tmp/_probe_latlon_halo_census.py-49-
scripts/tmp/_probe_latlon_halo_census.py-50-model = bench._build_model(n_lat, n_lon, nlev)
scripts/tmp/_probe_latlon_halo_census.py-51-c = build_sharded_held_suarez_state_atm_latlon(model.grid, model.sigma_coord, mesh)
scripts/tmp/_probe_latlon_halo_census.py-52-step = make_sharded_atm_latlon_step(model, mesh, physics_fn=None)
scripts/tmp/_probe_latlon_halo_census.py-53-
scripts/tmp/_probe_latlon_halo_census.py-54-census = hlo_collective_census(lambda s: step(s, 60.0), c)
scripts/tmp/_probe_latlon_halo_census.py-55-
scripts/tmp/_probe_latlon_halo_census.py:56:# Exact per-CP payload: sum collective-permute operand bytes from the
scripts/tmp/_probe_latlon_halo_census.py-57-# compiled HLO text (kills the row-slab lower-bound approximation).
scripts/tmp/_probe_latlon_halo_census.py-58-import re
scripts/tmp/_probe_latlon_halo_census.py-59-lowered = jax.jit(lambda s: step(s, 60.0)).lower(c).compile()
scripts/tmp/_probe_latlon_halo_census.py-60-txt = lowered.as_text()
scripts/tmp/_probe_latlon_halo_census.py-61-DT = {"f32": 4, "f64": 8, "bf16": 2, "f16": 2, "s32": 4, "u32": 4,
scripts/tmp/_probe_latlon_halo_census.py-62-      "pred": 1, "s8": 1, "u8": 1, "c64": 8, "c128": 16}
scripts/tmp/_probe_latlon_halo_census.py-63-cp_bytes = 0
scripts/tmp/_probe_latlon_halo_census.py-64-cp_ops = 0
scripts/tmp/_probe_latlon_halo_census.py:65:for m in re.finditer(r"collective-permute(?:-start)?\(", txt):
scripts/tmp/_probe_latlon_halo_census.py:66:    # walk back to the result shape at line start: '%name = TYPE[dims]{...} collective-permute'
scripts/tmp/_probe_latlon_halo_census.py-67-    line = txt[txt.rfind("\n", 0, m.start())+1:m.start()]
scripts/tmp/_probe_latlon_halo_census.py-68-    sm = re.search(r"(\w+)\[([0-9,]*)\]", line)
scripts/tmp/_probe_latlon_halo_census.py-69-    if not sm:
scripts/tmp/_probe_latlon_halo_census.py-70-        continue
scripts/tmp/_probe_latlon_halo_census.py-71-    dt, dims = sm.group(1), sm.group(2)
scripts/tmp/_probe_latlon_halo_census.py-72-    if dt not in DT:
scripts/tmp/_probe_latlon_halo_census.py-73-        continue
scripts/tmp/_probe_latlon_halo_census.py-74-    n = 1
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-436-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-437-    Contrast with driving ``jax.jit(step)`` in a Python loop (the prior
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-438-    production inner loop): ONE host dispatch per SEGMENT instead of per
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-439-    STEP, and the no-full-face-all-gather property holds for the WHOLE
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-440-    compiled program (gated on HLO text by
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-441-    ``tests/parallel/test_cube_tile_native_segment.py`` — zero
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-442-    ``all-gather``; collective counts n-INDEPENDENT past XLA's fixed
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-443-    boundary-iteration peel, i.e. the only collectives are the scan
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:444:    body's in-stage tile-halo ``collective-permute``s and, with
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-445-    ``fix_mass``, the fixer ``psum``).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-446-    The scanned body is the SAME blocked step — input layout == output
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-447-    layout (tile-sharded), so the carry never leaves tile layout and no
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-448-    D-grid<->cell conversion runs inside the scan (those live in
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-449-    ``enter``/``exit_`` only, once per segment boundary).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-450-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-451-    Carry dtype: leading steps are UNROLLED outside the ``lax.scan`` until
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-452-    the blocked state's dtype signature is a fixed point of the step
--
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py-302-    ``P(j, i) = q_r[j,i,k_sfc] · rho_total[j,i,k_sfc] · fall_speed``.
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py-303-
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py-304-    Convert to RCEMIP mm/day: ``P_mm_day = P [kg/m²/s] · 86400``
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py-305-    (using rho_water = 1000 kg/m³ → 1 mm = 1 kg/m²).
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py-306-
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py-307-    This is NOT a tracking of microphysical precipitation flux —
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py-308-    use the microphysics scheme's ``precipitation`` output for an
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py-309-    exact rate. Use this for quick sanity checks when the
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py:310:    microphysics output is not pipelined into the diagnostics path.
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py-311-    """
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py-312-    _validate_plane_state(state, height_coord)
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py-313-    _validate_slot(qr_slot, state.tracers.data.shape[-1], "qr_slot")
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py-314-    k_sfc = -1
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py-315-    q_r_sfc = state.tracers.data[..., k_sfc, qr_slot]
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py-316-    rho_sfc = (
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py-317-        height_coord.rho_ref[k_sfc]
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py-318-        + state.rho_prime.data[..., k_sfc]
--
packages/core/legoesm/parallel/early_init.py-176-    (``libnccl-net*``) is discoverable on ``LD_LIBRARY_PATH``/``LD_PRELOAD``
packages/core/legoesm/parallel/early_init.py-177-    / ``NCCL_NET_PLUGIN``.  ``missing_net_plugin_multi_node`` records the
packages/core/legoesm/parallel/early_init.py-178-    FACT of a multi-node launch with no net plugin visible — on OFI fabrics
packages/core/legoesm/parallel/early_init.py-179-    (Derecho Slingshot) that means NCCL silently runs correct-but-slow TCP
packages/core/legoesm/parallel/early_init.py-180-    sockets (git cba9715b2: 'route-B NCCL works cross-node but
packages/core/legoesm/parallel/early_init.py-181-    socket-bound'); native-IB fabrics run fine without a plugin, which is
packages/core/legoesm/parallel/early_init.py-182-    why the field states the fact, not the inference.  Advisory (the
packages/core/legoesm/parallel/early_init.py-183-    definitive check stays ``NCCL_DEBUG=INFO`` in the job log); recorded so
packages/core/legoesm/parallel/early_init.py:184:    a socket-bound row is falsifiable from the record.
packages/core/legoesm/parallel/early_init.py-185-    """
packages/core/legoesm/parallel/early_init.py-186-    import glob
packages/core/legoesm/parallel/early_init.py-187-
packages/core/legoesm/parallel/early_init.py-188-    plugin_hit = None
packages/core/legoesm/parallel/early_init.py-189-    if os.environ.get("NCCL_NET_PLUGIN"):
packages/core/legoesm/parallel/early_init.py-190-        plugin_hit = os.environ["NCCL_NET_PLUGIN"]
packages/core/legoesm/parallel/early_init.py-191-    else:
packages/core/legoesm/parallel/early_init.py-192-        # LD_PRELOAD entries are separated by SPACES or colons (ld.so(8));
--
packages/core/legoesm/parallel/cubesphere_exchange.py-1840-    multi-device scaling)."""
packages/core/legoesm/parallel/cubesphere_exchange.py-1841-    bad = find_fullcube_allgathers(hlo_text, n_faces=n_faces, n=n)
packages/core/legoesm/parallel/cubesphere_exchange.py-1842-    if bad:
packages/core/legoesm/parallel/cubesphere_exchange.py-1843-        raise RuntimeError(
packages/core/legoesm/parallel/cubesphere_exchange.py-1844-            f"{context}: compiled HLO contains {len(bad)} all-gather "
packages/core/legoesm/parallel/cubesphere_exchange.py-1845-            f"op(s) with full-cube face extent {bad[:8]} — the SPMD halo "
packages/core/legoesm/parallel/cubesphere_exchange.py-1846-            f"is materializing all {n_faces} faces per device (compute "
packages/core/legoesm/parallel/cubesphere_exchange.py-1847-            f"replication, HLO probe job 8456476).  Expected the "
packages/core/legoesm/parallel/cubesphere_exchange.py:1848:            f"ppermute multiface exchange (collective-permute only).  "
packages/core/legoesm/parallel/cubesphere_exchange.py-1849-            f"If the all_gather diagnostic backend was intended, set "
packages/core/legoesm/parallel/cubesphere_exchange.py-1850-            f"LEGOESM_SPMD_FORCE_ALLGATHER=1 explicitly."
packages/core/legoesm/parallel/cubesphere_exchange.py-1851-        )
packages/core/legoesm/parallel/cubesphere_exchange.py-1852-
packages/core/legoesm/parallel/cubesphere_exchange.py-1853-
packages/core/legoesm/parallel/cubesphere_exchange.py-1854-# ===================================================================
packages/core/legoesm/parallel/cubesphere_exchange.py-1855-# Public scalar exchange API
packages/core/legoesm/parallel/cubesphere_exchange.py-1856-# ===================================================================
--
packages/core/legoesm/parallel/sharded_dynamics.py-719-    (the tiled path is unvalidated — bench guards exclude it).
packages/core/legoesm/parallel/sharded_dynamics.py-720-    """
packages/core/legoesm/parallel/sharded_dynamics.py-721-    if config.mesh is None:
packages/core/legoesm/parallel/sharded_dynamics.py-722-        logger.info("make_sharded_step: single-device mode, using plain JIT")
packages/core/legoesm/parallel/sharded_dynamics.py-723-        return _SingleDeviceStep(model)
packages/core/legoesm/parallel/sharded_dynamics.py-724-
packages/core/legoesm/parallel/sharded_dynamics.py-725-    # Activate explicit SPMD halo exchange for face-sharded cubed-sphere.
packages/core/legoesm/parallel/sharded_dynamics.py-726-    # This replaces implicit cross-shard reads with explicit
packages/core/legoesm/parallel/sharded_dynamics.py:727:    # collective-permute rounds, producing much better XLA communication
packages/core/legoesm/parallel/sharded_dynamics.py-728-    # patterns.
packages/core/legoesm/parallel/sharded_dynamics.py-729-    #
packages/core/legoesm/parallel/sharded_dynamics.py-730-    # Iter-49 generalised activation from "exactly 6 devices" to "any
packages/core/legoesm/parallel/sharded_dynamics.py-731-    # divisor of 6" (1, 2, 3, 6) on the allgather kernels; the
packages/core/legoesm/parallel/sharded_dynamics.py-732-    # ppermute-multiface refit then made ppermute the DEFAULT exchange
packages/core/legoesm/parallel/sharded_dynamics.py-733-    # for every face-sharded count and at halo=2 — the allgather
packages/core/legoesm/parallel/sharded_dynamics.py-734-    # variant provably replicated ALL compute per device (HLO probe job
packages/core/legoesm/parallel/sharded_dynamics.py-735-    # 8456476, per-device FLOPs ratio 1.00 at 2 devices) and is now an
--
packages/core/legoesm/parallel/halo_exchange.py-45-   is not possible at this level.
packages/core/legoesm/parallel/halo_exchange.py-46-
packages/core/legoesm/parallel/halo_exchange.py-47-2. **Face-only mode uses neighbor sendrecv**: When rank count ≤ 6,
packages/core/legoesm/parallel/halo_exchange.py-48-   edges are grouped by neighbor rank and exchanged via one sendrecv
packages/core/legoesm/parallel/halo_exchange.py-49-   per unique neighbor (≤4 messages).  For >6 ranks, the tiled mode
packages/core/legoesm/parallel/halo_exchange.py-50-   uses point-to-point sendrecv with only actual neighbors.
packages/core/legoesm/parallel/halo_exchange.py-51-
packages/core/legoesm/parallel/halo_exchange.py-52-3. **Tiled mode is sequential per edge**: The 4 edge directions are
packages/core/legoesm/parallel/halo_exchange.py:53:   exchanged in sequence (not pipelined).  Each sendrecv blocks until
packages/core/legoesm/parallel/halo_exchange.py-54-   both send and receive complete.
packages/core/legoesm/parallel/halo_exchange.py-55-
packages/core/legoesm/parallel/halo_exchange.py-56-Near-term improvement path:
packages/core/legoesm/parallel/halo_exchange.py-57-  - Batch edge packing into a single contiguous buffer per neighbor
packages/core/legoesm/parallel/halo_exchange.py-58-    rank and do one sendrecv per neighbor (reduces from 4 to ≤4
packages/core/legoesm/parallel/halo_exchange.py-59-    messages per rank, with larger messages for better bandwidth).
packages/core/legoesm/parallel/halo_exchange.py-60-  - When ``mpi4jax`` gains ``Isend``/``Irecv``, convert to non-blocking
packages/core/legoesm/parallel/halo_exchange.py-61-    with ``Waitall`` after all sends/receives are posted.
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
scripts/tmp/probe_ocean_halo_bytes.py-39-s = shard_state_latlon(s0, mesh)
scripts/tmp/probe_ocean_halo_bytes.py-40-
scripts/tmp/probe_ocean_halo_bytes.py-41-txt = jax.jit(lambda st: step(st, 600.0)).lower(s).compile().as_text()
scripts/tmp/probe_ocean_halo_bytes.py-42-
scripts/tmp/probe_ocean_halo_bytes.py-43-# Sum the byte size of every collective's operand shape.
scripts/tmp/probe_ocean_halo_bytes.py-44-DTYPE_BYTES = {"f32": 4, "f64": 8, "s32": 4, "pred": 1, "bf16": 2, "f16": 2}
scripts/tmp/probe_ocean_halo_bytes.py-45-tot = {}
scripts/tmp/probe_ocean_halo_bytes.py-46-pat = re.compile(r"=\s*\(?([a-z0-9]+)\[([0-9,]*)\][^=]*?\b"
scripts/tmp/probe_ocean_halo_bytes.py:47:                 r"(collective-permute|all-reduce|all-gather|all-to-all)\b")
scripts/tmp/probe_ocean_halo_bytes.py-48-for m in pat.finditer(txt):
scripts/tmp/probe_ocean_halo_bytes.py-49-    dt, dims, op = m.group(1), m.group(2), m.group(3)
scripts/tmp/probe_ocean_halo_bytes.py-50-    n = 1
scripts/tmp/probe_ocean_halo_bytes.py-51-    for d in [d for d in dims.split(",") if d]:
scripts/tmp/probe_ocean_halo_bytes.py-52-        n *= int(d)
scripts/tmp/probe_ocean_halo_bytes.py-53-    b = n * DTYPE_BYTES.get(dt, 4)
scripts/tmp/probe_ocean_halo_bytes.py-54-    tot.setdefault(op, [0, 0])
scripts/tmp/probe_ocean_halo_bytes.py-55-    tot[op][0] += 1

exec
/usr/bin/bash -lc "sed -n '1,260p' scripts/tmp/_probe_latlon_halo_census.py
rg -n -C 10 \"def calibrated_bound|calibrated_bound\" packages scripts docs/performance/scaling --glob '*.py' --glob '*.sbatch' --glob '*.md'
rg -n -C 8 \"26630370|26630438|combine-8MB|combine.*8|pipelined-p2p\" scripts/tmp docs/performance/scaling --glob '*.sbatch' --glob '*.py' --glob '*.md'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 483ms:
"""Virtual-device HLO census of the atm lat-lon SPMD band step.

Feeds the 'no analytic halo-message census for the atm latlon step yet'
bound gap (audit item 4 follow-up): count collectives + estimate bytes
per step from the COMPILED HLO on a forced-host-platform device mesh,
using the bench's own builders (no re-derived model).

CAVEAT (metadata.py:214): counts are flag- and backend-dependent — the
default schedule reproduces from a CPU virtual-device compile, but
GPU-only collective combining can LOWER the executed count. So counts
here are census=virtual-cpu; a bound built on them is a MODEL.
Structure check across nd: per-device collective count must be
nd-independent for a 1-D band halo.

Usage: python _probe_latlon_halo_census.py ND N_LAT [N_LON] [NLEV]
"""
import os
import sys

nd = int(sys.argv[1]) if len(sys.argv) > 1 else 8
n_lat = int(sys.argv[2]) if len(sys.argv) > 2 else 512
n_lon = int(sys.argv[3]) if len(sys.argv) > 3 else 2 * n_lat
nlev = int(sys.argv[4]) if len(sys.argv) > 4 else 26

os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                           + f" --xla_force_host_platform_device_count={nd}")
os.environ["JAX_PLATFORMS"] = "cpu"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bench"))

import importlib
import json

import jax
import numpy as np

bench = importlib.import_module("bench_atm_latlon_spmd_scaling")
from metadata import hlo_collective_census  # noqa: E402

from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (  # noqa: E402
    build_sharded_held_suarez_state_atm_latlon,
    make_sharded_atm_latlon_step,
)

devs = jax.devices()
assert len(devs) == nd, (len(devs), nd)
mesh = jax.sharding.Mesh(np.array(devs), axis_names=("lat",))

model = bench._build_model(n_lat, n_lon, nlev)
c = build_sharded_held_suarez_state_atm_latlon(model.grid, model.sigma_coord, mesh)
step = make_sharded_atm_latlon_step(model, mesh, physics_fn=None)

census = hlo_collective_census(lambda s: step(s, 60.0), c)

# Exact per-CP payload: sum collective-permute operand bytes from the
# compiled HLO text (kills the row-slab lower-bound approximation).
import re
lowered = jax.jit(lambda s: step(s, 60.0)).lower(c).compile()
txt = lowered.as_text()
DT = {"f32": 4, "f64": 8, "bf16": 2, "f16": 2, "s32": 4, "u32": 4,
      "pred": 1, "s8": 1, "u8": 1, "c64": 8, "c128": 16}
cp_bytes = 0
cp_ops = 0
for m in re.finditer(r"collective-permute(?:-start)?\(", txt):
    # walk back to the result shape at line start: '%name = TYPE[dims]{...} collective-permute'
    line = txt[txt.rfind("\n", 0, m.start())+1:m.start()]
    sm = re.search(r"(\w+)\[([0-9,]*)\]", line)
    if not sm:
        continue
    dt, dims = sm.group(1), sm.group(2)
    if dt not in DT:
        continue
    n = 1
    for d in dims.split(","):
        if d:
            n *= int(d)
    cp_bytes += n * DT[dt]
    cp_ops += 1
out = {
    "cp_ops_with_shape": cp_ops,
    "cp_result_bytes_per_dev_per_step": cp_bytes,
    "nd": nd, "n_lat": n_lat, "n_lon": n_lon, "nlev": nlev,
    "rows_per_dev": n_lat // nd,
    "census_backend": "virtual-cpu (forced host platform)",
    "census": census,
}
print(json.dumps(out))
scripts/bench/bench_ocean_latlon_spmd_scaling.py-51-# Sibling-script import (ocean_invariants / conservation helpers reuse —
scripts/bench/bench_ocean_latlon_spmd_scaling.py-52-# same pattern as bench_ocean_mpi_scaling's own cross-script imports).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-53-sys.path.insert(0, str(Path(__file__).parent))
scripts/bench/bench_ocean_latlon_spmd_scaling.py-54-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-55-# Shared self-describing scaling metadata (anti-fake-scaling audit): merged
scripts/bench/bench_ocean_latlon_spmd_scaling.py-56-# under rec["metadata"] so a virtual-CPU-device proxy, a gloo/TCP fabric run,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-57-# or an f32 ablation is falsifiable from the JSONL row alone.  metadata.py
scripts/bench/bench_ocean_latlon_spmd_scaling.py-58-# imports JAX lazily, so this is safe before jax.distributed.initialize.
scripts/bench/bench_ocean_latlon_spmd_scaling.py-59-from metadata import (  # noqa: E402
scripts/bench/bench_ocean_latlon_spmd_scaling.py-60-    annotate_incomplete,
scripts/bench/bench_ocean_latlon_spmd_scaling.py:61:    calibrated_bound,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-62-    comm_accounting,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-63-    scaling_metadata,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-64-    tidy_throughput_fields,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-65-    wet_cell_metrics,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-66-)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-67-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-68-# SPMD full-step parity tolerances — the FLOATING-POINT RE-ASSOCIATION floor
scripts/bench/bench_ocean_latlon_spmd_scaling.py-69-# of the sharded split-explicit barotropic (ppermute/psum reduction-order
scripts/bench/bench_ocean_latlon_spmd_scaling.py-70-# change over the ~30-substep loop, pole-amplified), NOT a bug margin; a real
scripts/bench/bench_ocean_latlon_spmd_scaling.py-71-# missing-halo regression shows up at O(1e-3+) at the band cuts.  Values
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-600-            final_global, args.dt, model.grid, model.z_coord, _probe_cfg,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-601-            return_residual=True)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-602-        zero_forcing_probe_residual = float(
scripts/bench/bench_ocean_latlon_spmd_scaling.py-603-            jax.block_until_ready(_probe_out[2]))
scripts/bench/bench_ocean_latlon_spmd_scaling.py-604-        zero_forcing_probe_measured = True
scripts/bench/bench_ocean_latlon_spmd_scaling.py-605-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-606-    # --- Communication accounting (audit item 4) + calibrated bound (8) ----
scripts/bench/bench_ocean_latlon_spmd_scaling.py-607-    # Analytic INTER-DEVICE census, barotropic-solver scope ONLY (the
scripts/bench/bench_ocean_latlon_spmd_scaling.py-608-    # baroclinic 3-D pads are not counted -> bytes/comm are a LOWER census,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-609-    # flagged machine-readably via halo_bytes_is_lower_bound; T_bound is a
scripts/bench/bench_ocean_latlon_spmd_scaling.py:610:    # heuristic model, see calibrated_bound's docstring).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-611-    # implicit_cn PCG: each Helmholtz apply pads eta N+S (gradient stencil)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-612-    # + the v-face flux row (divergence) ~= 2 exchanges/apply, applied
scripts/bench/bench_ocean_latlon_spmd_scaling.py-613-    # iters + 1 times (incl. the initial residual); reductions = the dot
scripts/bench/bench_ocean_latlon_spmd_scaling.py-614-    # batches (2/iter standard, 1/iter single_reduce) + the initial batch
scripts/bench/bench_ocean_latlon_spmd_scaling.py-615-    # + the mass-projection psum + the eta-floor clamp psum.
scripts/bench/bench_ocean_latlon_spmd_scaling.py-616-    # The split-explicit substep-pad ESTIMATOR is only meaningful for
scripts/bench/bench_ocean_latlon_spmd_scaling.py-617-    # explicit_substep — implicit_cn has no substep loop, so publishing its
scripts/bench/bench_ocean_latlon_spmd_scaling.py-618-    # numbers on implicit rows would mislabel their traffic (codex batch4).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-619-    if args.baro_solver == "explicit_substep":
scripts/bench/bench_ocean_latlon_spmd_scaling.py-620-        from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-672-    # facing name but now carries the fused number; the individually-synced
scripts/bench/bench_ocean_latlon_spmd_scaling.py-673-    # dispatch latency is reported separately as ``step_latency_ms``.
scripts/bench/bench_ocean_latlon_spmd_scaling.py-674-    # None on the zero-length parity path (--parity-gate --steps 1): a
scripts/bench/bench_ocean_latlon_spmd_scaling.py-675-    # zero-step block has no per-step time — nulls propagate honestly.
scripts/bench/bench_ocean_latlon_spmd_scaling.py-676-    med = timing["fused_step_ms"]
scripts/bench/bench_ocean_latlon_spmd_scaling.py-677-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-678-    # Calibrated T_bound (audit item 8): nd=1 rows ARE their own compute
scripts/bench/bench_ocean_latlon_spmd_scaling.py-679-    # ingredient; nd>1 rows need the nd=1 fused number passed in (else the
scripts/bench/bench_ocean_latlon_spmd_scaling.py-680-    # bound is emitted null + flagged).  launch_host_ms=0: per-step launch
scripts/bench/bench_ocean_latlon_spmd_scaling.py-681-    # cost inside a fused lax.scan block is amortized to ~0.
scripts/bench/bench_ocean_latlon_spmd_scaling.py:682:    bound_rec = calibrated_bound(
scripts/bench/bench_ocean_latlon_spmd_scaling.py-683-        measured_fused_step_ms=med,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-684-        single_device_fused_step_ms=(med if nd == 1
scripts/bench/bench_ocean_latlon_spmd_scaling.py-685-                                     else args.single_dev_fused_ms),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-686-        halo_messages_per_step=comm_rec["halo_messages_per_step"],
scripts/bench/bench_ocean_latlon_spmd_scaling.py-687-        halo_bytes_per_step=comm_rec["halo_bytes_per_step"],
scripts/bench/bench_ocean_latlon_spmd_scaling.py-688-        n_reductions_per_step=_n_reductions,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-689-        rank_imbalance=float(timing["rank_imbalance"]),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-690-        latency_us=args.comm_latency_us,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-691-        bandwidth_GBs=args.comm_bandwidth_gbs,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-692-    )
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-71-# Bench dir for the shared metadata module (sibling-script import pattern —
scripts/bench/bench_atm_latlon_spmd_scaling.py-72-# needed when this file is loaded by path from tests, not run as a script).
scripts/bench/bench_atm_latlon_spmd_scaling.py-73-sys.path.insert(0, str(Path(__file__).resolve().parent))
scripts/bench/bench_atm_latlon_spmd_scaling.py-74-
scripts/bench/bench_atm_latlon_spmd_scaling.py-75-# Shared self-describing scaling metadata (anti-fake-scaling audit): merged
scripts/bench/bench_atm_latlon_spmd_scaling.py-76-# under rec["metadata"] so a virtual-CPU-device proxy, a gloo/TCP fabric run,
scripts/bench/bench_atm_latlon_spmd_scaling.py-77-# or an f32 ablation is falsifiable from the JSONL row alone.  metadata.py
scripts/bench/bench_atm_latlon_spmd_scaling.py-78-# imports JAX lazily, so this is safe before jax.distributed.initialize.
scripts/bench/bench_atm_latlon_spmd_scaling.py-79-from metadata import (  # noqa: E402
scripts/bench/bench_atm_latlon_spmd_scaling.py-80-    annotate_incomplete,
scripts/bench/bench_atm_latlon_spmd_scaling.py:81:    calibrated_bound,
scripts/bench/bench_atm_latlon_spmd_scaling.py-82-    comm_accounting,
scripts/bench/bench_atm_latlon_spmd_scaling.py-83-    scaling_metadata,
scripts/bench/bench_atm_latlon_spmd_scaling.py-84-    tidy_throughput_fields,
scripts/bench/bench_atm_latlon_spmd_scaling.py-85-)
scripts/bench/bench_atm_latlon_spmd_scaling.py-86-
scripts/bench/bench_atm_latlon_spmd_scaling.py-87-
scripts/bench/bench_atm_latlon_spmd_scaling.py-88-def _build_model(n_lat, n_lon, nlev):
scripts/bench/bench_atm_latlon_spmd_scaling.py-89-    from legoesm import constants
scripts/bench/bench_atm_latlon_spmd_scaling.py-90-    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
scripts/bench/bench_atm_latlon_spmd_scaling.py-91-        CGridLatLonPrimitiveEquationConfig, CGridLatLonPrimitiveEquationModel)
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-336-        _comm_note = ("no analytic halo-message census for the atm latlon "
scripts/bench/bench_atm_latlon_spmd_scaling.py-337-                      "step yet (audit item 4 follow-up) — comm fields null, "
scripts/bench/bench_atm_latlon_spmd_scaling.py-338-                      "not fabricated")
scripts/bench/bench_atm_latlon_spmd_scaling.py-339-    comm_rec = comm_accounting(
scripts/bench/bench_atm_latlon_spmd_scaling.py-340-        halo_messages_per_step=_msgs,
scripts/bench/bench_atm_latlon_spmd_scaling.py-341-        bytes_per_message=_bytes_msg,
scripts/bench/bench_atm_latlon_spmd_scaling.py-342-        full_state_gathers_per_step=0,   # fused scan/segment: no per-step gather
scripts/bench/bench_atm_latlon_spmd_scaling.py-343-        scope_note=_comm_note,
scripts/bench/bench_atm_latlon_spmd_scaling.py-344-        bytes_are_lower_bound=_bytes_lower,
scripts/bench/bench_atm_latlon_spmd_scaling.py-345-    )
scripts/bench/bench_atm_latlon_spmd_scaling.py:346:    bound_rec = calibrated_bound(
scripts/bench/bench_atm_latlon_spmd_scaling.py-347-        measured_fused_step_ms=_measured_med,
scripts/bench/bench_atm_latlon_spmd_scaling.py-348-        # Invalid rows feed the bound NOTHING: even the CLI-provided nd=1
scripts/bench/bench_atm_latlon_spmd_scaling.py-349-        # baseline is withheld so bound_ingredients.compute_ms cannot dress
scripts/bench/bench_atm_latlon_spmd_scaling.py-350-        # a diverging row up as a modelled one (codex).
scripts/bench/bench_atm_latlon_spmd_scaling.py-351-        single_device_fused_step_ms=(
scripts/bench/bench_atm_latlon_spmd_scaling.py-352-            _measured_med if nd == 1
scripts/bench/bench_atm_latlon_spmd_scaling.py-353-            else (args.single_dev_fused_ms if valid else None)),
scripts/bench/bench_atm_latlon_spmd_scaling.py-354-        halo_messages_per_step=comm_rec["halo_messages_per_step"],
scripts/bench/bench_atm_latlon_spmd_scaling.py-355-        halo_bytes_per_step=comm_rec["halo_bytes_per_step"],
scripts/bench/bench_atm_latlon_spmd_scaling.py-356-        n_reductions_per_step=_nred,
--
scripts/bench/metadata.py-616-        mcells_per_s = 0.0
scripts/bench/metadata.py-617-    return {
scripts/bench/metadata.py-618-        "dt_seconds": float(dt_seconds),
scripts/bench/metadata.py-619-        "time_per_step_ms": float(time_per_step_ms),
scripts/bench/metadata.py-620-        "total_cells": int(total_cells),
scripts/bench/metadata.py-621-        "sypd": sypd,
scripts/bench/metadata.py-622-        "mcells_per_s": mcells_per_s,
scripts/bench/metadata.py-623-    }
scripts/bench/metadata.py-624-
scripts/bench/metadata.py-625-
scripts/bench/metadata.py:626:#: Placeholder comm-fabric numbers for :func:`calibrated_bound` when the
scripts/bench/metadata.py-627-#: caller passes no measured values.  Ballpark single-node GPU-interconnect
scripts/bench/metadata.py-628-#: figures (order NVLink/PCIe), NOT measurements of THIS machine —
scripts/bench/metadata.py-629-#: MACHINE-CALIBRATED-REQUIRED: any bound built on them is emitted with
scripts/bench/metadata.py-630-#: ``bound_calibrated=False`` and must never be quoted as a hardware
scripts/bench/metadata.py-631-#: roofline.  Calibrate with a ping-pong / allreduce microbenchmark on the
scripts/bench/metadata.py-632-#: actual fabric and pass ``latency_us`` / ``bandwidth_GBs`` explicitly.
scripts/bench/metadata.py-633-DEFAULT_COMM_LATENCY_US = 25.0
scripts/bench/metadata.py-634-DEFAULT_COMM_BANDWIDTH_GBS = 10.0
scripts/bench/metadata.py-635-
scripts/bench/metadata.py-636-
--
scripts/bench/metadata.py-744-               for w in wet_columns_per_device]
scripts/bench/metadata.py-745-        if len(per) != int(n_devices):
scripts/bench/metadata.py-746-            raise ValueError(
scripts/bench/metadata.py-747-                "wet_cell_metrics: wet_columns_per_device has "
scripts/bench/metadata.py-748-                f"{len(per)} entries for n_devices={n_devices}")
scripts/bench/metadata.py-749-        out["wet_cell_levels_per_device_min"] = min(per)
scripts/bench/metadata.py-750-        out["wet_cell_levels_per_device_max"] = max(per)
scripts/bench/metadata.py-751-    return out
scripts/bench/metadata.py-752-
scripts/bench/metadata.py-753-
scripts/bench/metadata.py:754:def calibrated_bound(
scripts/bench/metadata.py-755-    *,
scripts/bench/metadata.py-756-    measured_fused_step_ms: float | None = None,
scripts/bench/metadata.py-757-    single_device_fused_step_ms: float | None = None,
scripts/bench/metadata.py-758-    halo_messages_per_step: int | None = None,
scripts/bench/metadata.py-759-    halo_bytes_per_step: int | None = None,
scripts/bench/metadata.py-760-    n_reductions_per_step: int | None = None,
scripts/bench/metadata.py-761-    rank_imbalance: float | None = None,
scripts/bench/metadata.py-762-    latency_us: float | None = None,
scripts/bench/metadata.py-763-    bandwidth_GBs: float | None = None,
scripts/bench/metadata.py-764-    launch_host_ms: float = 0.0,
--
scripts/bench/metadata.py-806-    ``bound_incomplete_reason``).  Any missing ingredient ->
scripts/bench/metadata.py-807-    ``t_bound_ms=None`` + ``bound_incomplete_reason`` naming it — an
scripts/bench/metadata.py-808-    incomplete bound is reported as incomplete, never invented.
scripts/bench/metadata.py-809-    """
scripts/bench/metadata.py-810-    calibrated = latency_us is not None and bandwidth_GBs is not None
scripts/bench/metadata.py-811-    lat_us = (DEFAULT_COMM_LATENCY_US if latency_us is None
scripts/bench/metadata.py-812-              else float(latency_us))
scripts/bench/metadata.py-813-    bw_gbs = (DEFAULT_COMM_BANDWIDTH_GBS if bandwidth_GBs is None
scripts/bench/metadata.py-814-              else float(bandwidth_GBs))
scripts/bench/metadata.py-815-    if lat_us < 0.0:
scripts/bench/metadata.py:816:        raise ValueError(f"calibrated_bound: latency_us must be >= 0, "
scripts/bench/metadata.py-817-                         f"got {lat_us}")
scripts/bench/metadata.py-818-    if bw_gbs <= 0.0:
scripts/bench/metadata.py:819:        raise ValueError(f"calibrated_bound: bandwidth_GBs must be > 0, "
scripts/bench/metadata.py-820-                         f"got {bw_gbs}")
scripts/bench/metadata.py-821-
scripts/bench/metadata.py-822-    missing = [name for name, v in (
scripts/bench/metadata.py-823-        ("single_device_fused_step_ms", single_device_fused_step_ms),
scripts/bench/metadata.py-824-        ("halo_messages_per_step", halo_messages_per_step),
scripts/bench/metadata.py-825-        ("halo_bytes_per_step", halo_bytes_per_step),
scripts/bench/metadata.py-826-        ("n_reductions_per_step", n_reductions_per_step),
scripts/bench/metadata.py-827-    ) if v is None]
scripts/bench/metadata.py-828-    # Placeholder fabric numbers + nonzero comm/reduction work would put a
scripts/bench/metadata.py-829-    # fabricated latency/bandwidth INTO the bound: honest null instead.
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1804-  the 1-D band structure check). Exact CP payload from compiled-HLO
docs/performance/scaling/levante_campaign_2026-07-24.md-1805-  result shapes: 4,635,408 B/dev/step at n_lon=1024 L26 f32 = 1.06x the
docs/performance/scaling/levante_campaign_2026-07-24.md-1806-  single-row slab model; linear in n_lon (checked 1024 vs 2048, 0.07 %
docs/performance/scaling/levante_campaign_2026-07-24.md-1807-  residual) -> **18.5 MB/dev/step at n_lon=4096 f32**. CAVEAT: CPU
docs/performance/scaling/levante_campaign_2026-07-24.md-1808-  lowering; GPU-side collective combining could change the executed
docs/performance/scaling/levante_campaign_2026-07-24.md-1809-  count (metadata.py:214) — the bound is a MODEL.
docs/performance/scaling/levante_campaign_2026-07-24.md-1810-* **Same-tile nd=1 compute baselines** (job 26630370, roofline recipe):
docs/performance/scaling/levante_campaign_2026-07-24.md-1811-  16x4096 f32 1.659 ms, 32x4096 f32 2.837, 16x4096 f64 2.973.
docs/performance/scaling/levante_campaign_2026-07-24.md-1812-  Approximation, recorded: nd=1 includes pole tiles -> compute term
docs/performance/scaling/levante_campaign_2026-07-24.md-1813-  biased HIGH -> bound conservative.
docs/performance/scaling/levante_campaign_2026-07-24.md:1814:* **Calibrated bound** (`metadata.calibrated_bound`, measured fabric
docs/performance/scaling/levante_campaign_2026-07-24.md-1815-  constants: IB 26.3 us / 23.5 GB/s, NVLink 17.8 / 64.2):
docs/performance/scaling/levante_campaign_2026-07-24.md-1816-
docs/performance/scaling/levante_campaign_2026-07-24.md-1817-| row | measured | t_bound (IB) | measured/bound |
docs/performance/scaling/levante_campaign_2026-07-24.md-1818-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-1819-| LL2048@64 f32 | 6.732 | 2.863 | **2.35** |
docs/performance/scaling/levante_campaign_2026-07-24.md-1820-| LL2048@128 f32 | 5.577 | 1.848 | **3.02** |
docs/performance/scaling/levante_campaign_2026-07-24.md-1821-| LL2048@128 f64 | 9.602 | 2.999 | **3.20** |
docs/performance/scaling/levante_campaign_2026-07-24.md-1822-
docs/performance/scaling/levante_campaign_2026-07-24.md-1823-* **The lat-lon GPU panel sits ~2.4-3.2x ABOVE its modeled limit** — the
docs/performance/scaling/levante_campaign_2026-07-24.md-1824-  eff-0.60 strong leg is NOT close to the fabric+compute floor. Leading
docs/performance/scaling/spmd_message_census_2026-07-08.md-79-
docs/performance/scaling/spmd_message_census_2026-07-08.md-80-## MEASURED lane-T verdicts (Derecho, 2026-07-09, 8×A100 route-B NCCL,
docs/performance/scaling/spmd_message_census_2026-07-08.md-81-## same-allocation A/B — latlon LL512-class atm, ocean LL288-class)
docs/performance/scaling/spmd_message_census_2026-07-08.md-82-
docs/performance/scaling/spmd_message_census_2026-07-08.md-83-| arm | latlon ms/step | latlon SYPD | ocean ms/step | verdict |
docs/performance/scaling/spmd_message_census_2026-07-08.md-84-|---|---|---|---|---|
docs/performance/scaling/spmd_message_census_2026-07-08.md-85-| base | 6.48 | 25.4 | 33.09 | control |
docs/performance/scaling/spmd_message_census_2026-07-08.md-86-| fused (`LEGOESM_LATLON_SPMD_FUSED_HALO=1`) | 7.36 | 22.3 | 32.72 | **latlon −12 % — default stays OFF**; ocean +1 % (noise) |
docs/performance/scaling/spmd_message_census_2026-07-08.md:87:| xla (CP-combine 32 MiB + pipelined p2p) | 7.22 | 22.8 | — | **−10 % — not recommended as-is**; split the two flags in a follow-up arm before discarding |
docs/performance/scaling/spmd_message_census_2026-07-08.md-88-| pgle (`JAX_ENABLE_PGLE=true`) | **5.97** | **27.5** | — | **+8.5 % — the winner**; recommend per-run on route-B latlon lanes |
docs/performance/scaling/spmd_message_census_2026-07-08.md-89-
docs/performance/scaling/spmd_message_census_2026-07-08.md-90-Readings:
docs/performance/scaling/spmd_message_census_2026-07-08.md-91-1. **Fused multi-pad loses at this size/count**: −29 % messages, but each
docs/performance/scaling/spmd_message_census_2026-07-08.md-92-   message ~4× larger plus the pack/unpack concats — at LL512/np8 the
docs/performance/scaling/spmd_message_census_2026-07-08.md-93-   per-message latency saved is smaller than the copy overhead added.
docs/performance/scaling/spmd_message_census_2026-07-08.md-94-   The flag stays opt-in (it may still win at higher rank counts /
docs/performance/scaling/spmd_message_census_2026-07-08.md-95-   smaller per-rank tiles where latency dominates — re-A/B there before
docs/performance/scaling/spmd_message_census_2026-07-08.md-96-   discarding).
docs/performance/scaling/spmd_message_census_2026-07-08.md-97-2. **PGLE's profile-guided re-scheduling is the real overlap win** —
docs/performance/scaling/spmd_message_census_2026-07-08.md-98-   +8.5 % without touching the model.  Keep it per-run opt-in
docs/performance/scaling/spmd_message_census_2026-07-08.md-99-   (recompiles after the profiling runs; AOT-incompatible), and wire it
docs/performance/scaling/spmd_message_census_2026-07-08.md-100-   into the production route-B job env for latlon-class lanes.
docs/performance/scaling/spmd_message_census_2026-07-08.md:101:3. **The combined xla arm hurt**; pipelined-p2p and the CP-combiner need
docs/performance/scaling/spmd_message_census_2026-07-08.md-102-   separate arms to attribute (follow-up lane-T variant).
docs/performance/scaling/spmd_message_census_2026-07-08.md-103-4. Caveats: one size per component, one repeat — treat sub-5 % deltas as
docs/performance/scaling/spmd_message_census_2026-07-08.md-104-   noise; the cube base/xla arms and np16/24 rungs are still pending.
docs/performance/scaling/spmd_message_census_2026-07-08.md-105-
docs/performance/scaling/spmd_message_census_2026-07-08.md-106-## Consequences — what to run next on Derecho/Levante
docs/performance/scaling/spmd_message_census_2026-07-08.md-107-
docs/performance/scaling/spmd_message_census_2026-07-08.md-108-The count floor being (near-)reached in code moves the lever to the GPU
docs/performance/scaling/spmd_message_census_2026-07-08.md-109-runtime, in this order (now wired as **lane T** in
--
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-49-multicontroller, 8 GPU), lane D (ocean multicontroller, 8 GPU), lane E
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-50-(icosahedral/MPAS multicontroller, 6 GPU). Larger ladders: resubmit with
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-51-more nodes and `GPU_RANKS`/`LL_RES` etc. per the README.
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-52-
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-53-**Production knob (measured 2026-07-09, use it):** add
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-54-`JAX_ENABLE_PGLE=true JAX_PGLE_PROFILING_RUNS=3` to the environment of
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-55-lat-lon-class route-B lanes — measured **+8.5%** (5.97 vs 6.48 ms/step at
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-56-8 GPU). Do **not** set `LEGOESM_LATLON_SPMD_FUSED_HALO=1` (measured −12%
docs/performance/scaling/derecho_colleague_runbook_2026-07.md:57:on latlon at this size) and do not add the CP-combine/pipelined-p2p XLA
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-58-flags (measured −10% combined). Full A/B table:
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-59-`docs/performance/scaling/spmd_message_census_2026-07-08.md`.
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-60-
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-61-**Resolution floors (or the curves will look "broken" and are not):**
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-62-keep ≥ ~30k columns per GPU. Concretely: lat-lon LL512+ and icosahedral
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-63-L7+ for 8–16 GPUs; coarse grids (156 km latlon, 112 km ico, C48 cube) do
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-64-NOT strong-scale on GPUs anywhere — per-device saturation, expected.
docs/performance/scaling/derecho_colleague_runbook_2026-07.md-65-
--
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-93-   size (C96/L40, ≥30k cols/GPU). Yields the census ladder + per-phase halo BW,
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-94-   reduction latency, roofline, overlap — the WHERE, not just the SYPD.
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-95-2. **Ocean reduction-wall A/B** (the highest-value code lever, already
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-96-   selectable): re-run `bench_ocean_mpi_scaling.py` / the SPMD ocean bench with
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-97-   `--pcg-variant single_reduce` and with split-explicit +
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-98-   `barotropic_local_subcycle_clamp` at ≥16 ranks — the calculus flips toward
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-99-   the reduction-free path where per-message latency dominates.
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-100-3. **Lane-T GPU-runtime A/B** (already wired, `RUN_TUNE=1`): PGLE was the
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:101:   measured winner (+8.5 %); the XLA collective-permute-combine + pipelined-p2p
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-102-   arms still need SPLITTING to attribute (the combined arm hurt −10 %).
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-103-4. Per §1, cube >6 GPU needs the sub-face tiled production step (separate
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-104-   project); the count floor is otherwise reached in code.
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-105-
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-106-## 5. Verification
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-107-
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-108-- `count_collectives` / `hlo_collective_census`:
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-109-  `tests/bench/test_scaling_metadata.py` (all-family synthetic HLO + error-safe
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1802-  `make_sharded_atm_latlon_step`): **41 collective-permutes + 1
docs/performance/scaling/levante_campaign_2026-07-24.md-1803-  all-reduce per step**, nd-INDEPENDENT (identical at nd=8 and nd=16 —
docs/performance/scaling/levante_campaign_2026-07-24.md-1804-  the 1-D band structure check). Exact CP payload from compiled-HLO
docs/performance/scaling/levante_campaign_2026-07-24.md-1805-  result shapes: 4,635,408 B/dev/step at n_lon=1024 L26 f32 = 1.06x the
docs/performance/scaling/levante_campaign_2026-07-24.md-1806-  single-row slab model; linear in n_lon (checked 1024 vs 2048, 0.07 %
docs/performance/scaling/levante_campaign_2026-07-24.md-1807-  residual) -> **18.5 MB/dev/step at n_lon=4096 f32**. CAVEAT: CPU
docs/performance/scaling/levante_campaign_2026-07-24.md-1808-  lowering; GPU-side collective combining could change the executed
docs/performance/scaling/levante_campaign_2026-07-24.md-1809-  count (metadata.py:214) — the bound is a MODEL.
docs/performance/scaling/levante_campaign_2026-07-24.md:1810:* **Same-tile nd=1 compute baselines** (job 26630370, roofline recipe):
docs/performance/scaling/levante_campaign_2026-07-24.md-1811-  16x4096 f32 1.659 ms, 32x4096 f32 2.837, 16x4096 f64 2.973.
docs/performance/scaling/levante_campaign_2026-07-24.md-1812-  Approximation, recorded: nd=1 includes pole tiles -> compute term
docs/performance/scaling/levante_campaign_2026-07-24.md-1813-  biased HIGH -> bound conservative.
docs/performance/scaling/levante_campaign_2026-07-24.md-1814-* **Calibrated bound** (`metadata.calibrated_bound`, measured fabric
docs/performance/scaling/levante_campaign_2026-07-24.md-1815-  constants: IB 26.3 us / 23.5 GB/s, NVLink 17.8 / 64.2):
docs/performance/scaling/levante_campaign_2026-07-24.md-1816-
docs/performance/scaling/levante_campaign_2026-07-24.md-1817-| row | measured | t_bound (IB) | measured/bound |
docs/performance/scaling/levante_campaign_2026-07-24.md-1818-|---|---|---|---|
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1823-* **The lat-lon GPU panel sits ~2.4-3.2x ABOVE its modeled limit** — the
docs/performance/scaling/levante_campaign_2026-07-24.md-1824-  eff-0.60 strong leg is NOT close to the fabric+compute floor. Leading
docs/performance/scaling/levante_campaign_2026-07-24.md-1825-  PLAUSIBLE mechanism (uninstrumented): effective per-CP cost
docs/performance/scaling/levante_campaign_2026-07-24.md-1826-  (launch + schedule + stream sync) well above the raw 26 us fabric
docs/performance/scaling/levante_campaign_2026-07-24.md-1827-  latency across 41 dependency-chained exchanges — the arXiv:2607.16100
docs/performance/scaling/levante_campaign_2026-07-24.md-1828-  small-collective regime. The model itself notes the serialized-latency
docs/performance/scaling/levante_campaign_2026-07-24.md-1829-  vs overlap biases pull opposite ways; treat measured/bound as a
docs/performance/scaling/levante_campaign_2026-07-24.md-1830-  consistency diagnostic, not proven headroom.
docs/performance/scaling/levante_campaign_2026-07-24.md:1831:* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
docs/performance/scaling/levante_campaign_2026-07-24.md:1832:  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
docs/performance/scaling/levante_campaign_2026-07-24.md-1833-  same-job control, falsifiability block in the script. The ocean-lane
docs/performance/scaling/levante_campaign_2026-07-24.md-1834-  null for these flags was reduction-dominated — first atm test.
--
scripts/tmp/ocean_ab_xla.sbatch-4-#SBATCH --constraint=a100_80
scripts/tmp/ocean_ab_xla.sbatch-5-#SBATCH --nodes=1
scripts/tmp/ocean_ab_xla.sbatch-6-#SBATCH --ntasks-per-node=1
scripts/tmp/ocean_ab_xla.sbatch-7-#SBATCH --gpus-per-node=4
scripts/tmp/ocean_ab_xla.sbatch-8-#SBATCH --exclusive
scripts/tmp/ocean_ab_xla.sbatch-9-#SBATCH --time=01:15:00
scripts/tmp/ocean_ab_xla.sbatch-10-#SBATCH --output=ocean_ab_xla.%j.log
scripts/tmp/ocean_ab_xla.sbatch-11-# XLA/runtime split arms on the LL384 ocean ladder: CP-combine alone,
scripts/tmp/ocean_ab_xla.sbatch:12:# pipelined-p2p alone (Derecho lane-T measured them only COMBINED at -10%),
scripts/tmp/ocean_ab_xla.sbatch-13-# PGLE (+8.5% on Derecho), vmix-f32 + baroclinic-f32 mixed precision.
scripts/tmp/ocean_ab_xla.sbatch-14-set -uo pipefail
scripts/tmp/ocean_ab_xla.sbatch-15-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
scripts/tmp/ocean_ab_xla.sbatch-16-SCRIPT_DIR="${SUBMIT_DIR}/scripts/cluster/scaling_levante"
scripts/tmp/ocean_ab_xla.sbatch-17-[ -f "$SCRIPT_DIR/_env.sh" ] || { echo "submit from repo root" >&2; exit 1; }
scripts/tmp/ocean_ab_xla.sbatch-18-export JAX_PLATFORMS=cuda
scripts/tmp/ocean_ab_xla.sbatch-19-source "${SCRIPT_DIR}/_env.sh"
scripts/tmp/ocean_ab_xla.sbatch-20-cd "$REPO"

exec
/usr/bin/bash -lc "sed -n '754,1000p' scripts/bench/metadata.py
sed -n '280,385p' scripts/bench/bench_atm_latlon_spmd_scaling.py
ls -lt scripts/tmp | sed -n '1,100p'
rg -l \"26630438|combine-8MB|LL2048@128|latlon.*combine|CP-combin\" scripts scripts/tmp docs --glob '*.sbatch' --glob '*.py' --glob '*.md' | sort" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 354ms:
def calibrated_bound(
    *,
    measured_fused_step_ms: float | None = None,
    single_device_fused_step_ms: float | None = None,
    halo_messages_per_step: int | None = None,
    halo_bytes_per_step: int | None = None,
    n_reductions_per_step: int | None = None,
    rank_imbalance: float | None = None,
    latency_us: float | None = None,
    bandwidth_GBs: float | None = None,
    launch_host_ms: float = 0.0,
) -> dict[str, Any]:
    """Calibrated per-fused-step time MODEL (audit item 8).

        T_bound = max(compute, comm) + reduction + launch_host

    A heuristic roofline-STYLE model, NOT a guaranteed lower bound (codex
    batch4): the comm term serializes the per-message latency sum
    (overlapping/pipelined messages beat it), while a partial halo census
    (e.g. barotropic-only) UNDERcounts bytes — the two biases pull in
    opposite directions.  Use ``measured_over_bound`` as a consistency
    diagnostic, not as proven headroom.

    Ingredients (all MEASURABLE, none fabricated):

    - ``compute``   = single-device ``fused_step_ms`` at the SAME
      per-device size (the caller passes its nd=1 row; ``None`` -> the
      bound is emitted null and flagged incomplete).
    - ``comm``      = ``messages x latency + bytes / bandwidth`` — halo
      traffic, modeled as overlappable with compute, hence the ``max``.
    - ``reduction`` = ``n_reductions x latency`` — sequentially DEPENDENT
      allreduce-type collectives (CG dot products); latency-bound at
      bench scales, so bytes are neglected (small-message model).
    - ``launch_host`` — per-step dispatch overhead; ~0 inside a fused
      ``lax.scan`` block (amortized), so benches pass the default 0.0;
      drivers stepping one-at-a-time should pass their measured
      ``step_latency_ms - fused_step_ms``.

    ``rank_imbalance`` is reported as a DIAGNOSTIC ingredient
    (``imbalance_ms = (rank_imbalance - 1) x compute``) and deliberately
    NOT added to ``T_bound``: the measured max/median ratio already
    contains communication/reduction jitter, so adding it would
    double-count terms already modeled (codex batch4).

    ``latency_us`` / ``bandwidth_GBs`` default to the
    MACHINE-CALIBRATED-REQUIRED placeholders
    (:data:`DEFAULT_COMM_LATENCY_US` / :data:`DEFAULT_COMM_BANDWIDTH_GBS`);
    whenever either default is used the result carries
    ``bound_calibrated=False``.  Placeholders may only ever MULTIPLY ZERO
    work: with a nonzero communication/reduction census an uncalibrated
    fabric would fabricate a number, so the missing calibration is treated
    as a missing ingredient and the bound is emitted null (named in
    ``bound_incomplete_reason``).  Any missing ingredient ->
    ``t_bound_ms=None`` + ``bound_incomplete_reason`` naming it — an
    incomplete bound is reported as incomplete, never invented.
    """
    calibrated = latency_us is not None and bandwidth_GBs is not None
    lat_us = (DEFAULT_COMM_LATENCY_US if latency_us is None
              else float(latency_us))
    bw_gbs = (DEFAULT_COMM_BANDWIDTH_GBS if bandwidth_GBs is None
              else float(bandwidth_GBs))
    if lat_us < 0.0:
        raise ValueError(f"calibrated_bound: latency_us must be >= 0, "
                         f"got {lat_us}")
    if bw_gbs <= 0.0:
        raise ValueError(f"calibrated_bound: bandwidth_GBs must be > 0, "
                         f"got {bw_gbs}")

    missing = [name for name, v in (
        ("single_device_fused_step_ms", single_device_fused_step_ms),
        ("halo_messages_per_step", halo_messages_per_step),
        ("halo_bytes_per_step", halo_bytes_per_step),
        ("n_reductions_per_step", n_reductions_per_step),
    ) if v is None]
    # Placeholder fabric numbers + nonzero comm/reduction work would put a
    # fabricated latency/bandwidth INTO the bound: honest null instead.
    # (Zero work is fabric-independent — nd=1 rows keep their trivial
    # compute-only bound.)
    has_comm_work = any(
        v is not None and float(v) > 0.0
        for v in (halo_messages_per_step, halo_bytes_per_step,
                  n_reductions_per_step))
    if has_comm_work and not calibrated:
        if latency_us is None:
            missing.append("latency_us(placeholder with nonzero comm)")
        if bandwidth_GBs is None:
            missing.append("bandwidth_GBs(placeholder with nonzero comm)")

    compute_ms = (None if single_device_fused_step_ms is None
                  else float(single_device_fused_step_ms))
    comm_ms = None
    if halo_messages_per_step is not None and halo_bytes_per_step is not None:
        comm_ms = (float(halo_messages_per_step) * lat_us * 1e-3
                   + float(halo_bytes_per_step) / (bw_gbs * 1e9) * 1e3)
    reduction_ms = (None if n_reductions_per_step is None
                    else float(n_reductions_per_step) * lat_us * 1e-3)
    # Diagnostic ONLY — not a T_bound term (see the docstring).
    imbalance_ms = None
    if rank_imbalance is not None and compute_ms is not None:
        imbalance_ms = max(float(rank_imbalance) - 1.0, 0.0) * compute_ms

    if missing:
        t_bound_ms = None
        measured_over_bound = None
    else:
        t_bound_ms = (max(compute_ms, comm_ms) + reduction_ms
                      + float(launch_host_ms))
        measured_over_bound = (
            float(measured_fused_step_ms) / t_bound_ms
            if measured_fused_step_ms is not None and t_bound_ms > 0.0
            else None)

    return {
        "t_bound_ms": (None if t_bound_ms is None else round(t_bound_ms, 4)),
        "measured_over_bound": (None if measured_over_bound is None
                                else round(measured_over_bound, 4)),
        "bound_calibrated": bool(calibrated),
        "bound_incomplete_reason": (missing or None),
        "bound_ingredients": {
            "compute_ms": compute_ms,
            "comm_ms": (None if comm_ms is None else round(comm_ms, 6)),
            "reduction_ms": (None if reduction_ms is None
                             else round(reduction_ms, 6)),
            "imbalance_ms": (None if imbalance_ms is None
                             else round(imbalance_ms, 6)),
            "launch_host_ms": float(launch_host_ms),
            "latency_us": lat_us,
            "bandwidth_GBs": bw_gbs,
            "halo_messages_per_step": halo_messages_per_step,
            "halo_bytes_per_step": halo_bytes_per_step,
            "n_reductions_per_step": n_reductions_per_step,
            "rank_imbalance": rank_imbalance,
        },
    }


def timed_scan_blocks(
    advance,
    state,
    *,
    block_steps: int,
    n_blocks: int = 2,
    probe_steps: int = 3,
    sync_label: str = "timed_scan_blocks",
):
    """Measurement-contract timing: fused ``lax.scan`` blocks + probe latency.

    The trustworthy production-like number is a MULTI-STEP ``lax.scan`` block
    with device synchronization only AROUND the block (per-step host sync in a
    Python loop measures dispatch+sync latency, not fused device throughput —
    the audited anti-pattern in the SPMD benches).  Per-step dispatch latency
    is still physically meaningful (drivers that must step one-at-a-time pay
    it), so it is measured SEPARATELY by a short individually-synced probe and
    reported as ``step_latency_ms`` — never mixed into the fused number.

    Multi-controller runs additionally allgather EVERY process's full
    per-block vector and reduce per block: the parallel time of block ``b``
    is the SLOWEST process in that block.  A per-process median gathered
    alone would hide an alternating straggler (every rank's median can be
    fast even though every block has a slow rank — codex batch4).

    Parameters
    ----------
    advance
        ``advance(state) -> state`` — ONE production step with all static
        knobs (dt, forcing, ...) closed over.  May itself be jitted; it is
        re-traced INTO the fused scan (same graph, no double-jit penalty).
        MUST NOT donate its input buffers (``donate_argnums``): the scan
        pre-compile below runs on a SHALLOW pytree copy whose leaves ALIAS
        the live seed state — donation would invalidate the seed's buffers
        mid-benchmark.
    state
        Initial (already sharded, post-seed) model state pytree.
    block_steps
        Steps per fused ``lax.scan`` block (the amortizing window).
        ``>= 0``; ``0`` is the documented ZERO-LENGTH parity path (the
        block advances nothing and ``fused_step_ms`` is ``None`` — a
        zero-step block has no per-step time).
    n_blocks
        Timed blocks (``>= 1``); per-block times expose drift.
    probe_steps
        Individually host-synced steps (``>= 0``) for the separate
        dispatch-latency probe (each one costs a device round-trip).
    sync_label
        Base label for the multi-controller ``sync_global_devices`` fences.

    Invalid schedule values raise ``ValueError`` — never silently clamped,
    so the recorded schedule is ALWAYS the executed schedule.  Multi-process
    runs first allgather-verify the schedule tuple itself: processes that
    disagree on ``(block_steps, n_blocks, probe_steps)`` would enter
    DIFFERENT named-fence schedules and deadlock; the verification is the
    one collective every process reaches, so a mismatch raises everywhere.

    Returns
    -------
    (state, metrics) — final state (compile + probe + all blocks advanced)
    and a dict:
      ``compile_ms``           first-call cost of ``advance`` (trace+compile)
      ``scan_compile_ms``      first-call cost of the fused scan itself
      ``step_latency_ms``      median individually-synced per-step wall time
      ``block_ms``             per-block wall times, THIS process (list)
      ``parallel_block_ms``    per-block wall times of the PARALLEL step:
                               max over processes, per block (== ``block_ms``
                               for a single process)
      ``fused_step_ms``        median(parallel_block_ms)/block_steps — the
                               headline (``None`` when ``block_steps == 0``)
      ``rank_imbalance``       median over blocks of the per-block
                               max/median-across-processes ratio (>= 1.0;
                               exactly 1.0 for a single process)
      ``rank_imbalance_per_block``  the per-block ratios themselves
      ``block_steps``/``n_blocks``/``probe_steps``  the EXECUTED schedule
    """
    import time

    import jax
    import numpy as np

    multi = jax.process_count() > 1
    if multi:
        # Collectively verify the schedule BEFORE compilation or any
        # schedule-dependent fence: this allgather is the single collective
        # every process reaches first, so on a mismatch EVERY process sees
        # the same gathered table and raises together instead of hanging in
        # mismatched named fences (codex batch4 deadlock hazard).
        from jax.experimental import multihost_utils
        _sched = np.asarray(multihost_utils.process_allgather(
            np.array([block_steps, n_blocks, probe_steps], dtype=np.int64)))
        if not bool((_sched == _sched[0]).all()):
            raise ValueError(
                "timed_scan_blocks: processes disagree on the schedule "
                f"(block_steps, n_blocks, probe_steps) = {_sched.tolist()} "
                "per process — a mismatched schedule deadlocks in the "
                "named fences.")
    # No silent rewrites: invalid values raise; the recorded schedule IS
    # the executed schedule.  block_steps == 0 stays legal (the documented
    # zero-length parity path).
    if block_steps < 0:
        raise ValueError(
            f"timed_scan_blocks: block_steps must be >= 0, got {block_steps}")
    if n_blocks < 1:
        raise ValueError(
            f"timed_scan_blocks: n_blocks must be >= 1, got {n_blocks}")
    if probe_steps < 0:
        raise ValueError(
            f"timed_scan_blocks: probe_steps must be >= 0, got {probe_steps}")

    def _block(tree):
                      "valid=false) and its throughput fields are nulled")
                break
        completed_blocks = len(per_block_ms)
        per_step_ms = [b / seg_n for b in per_block_ms]
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices("atm_latlon_spmd_bench_end")
        steady = per_step_ms[args.warmup:]
        if not steady:
            # Divergence stopped the run inside the warmup window — fall back
            # to every completed unit (the record is already marked invalid;
            # this only keeps the diagnostic median well-defined).
            steady = per_step_ms
        med = float(np.median(steady))
    else:
        # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
        # blocks with sync only AROUND the block — the previous per-step
        # host-synced loop measured dispatch+sync latency, not fused device
        # throughput.  Dispatch latency stays measured SEPARATELY
        # (``step_latency_ms``); multi-controller runs record the
        # slowest-process block time + imbalance ratio.
        from metadata import timed_scan_blocks
        c, timing = timed_scan_blocks(
            lambda st: step(st, args.dt), c,
            block_steps=args.steps, n_blocks=args.blocks,
            probe_steps=args.probe_steps,
            sync_label="atm_latlon_spmd_bench")
        # Headline = fused per-step time from the SLOWEST process; key name
        # kept for the aggregators.
        med = float(timing["fused_step_ms"])

    # valid=false ONLY on an observed non-finite state; the default fused
    # lane (finite_ok=None: unchecked) stays valid.
    valid = finite_ok is not False
    # A diverging segment run's med is not a measurement: feed the bound
    # honest nulls (its flat throughput twins are nulled after assembly).
    _measured_med = med if valid else None
    # Honest per-device geometry residency (from the real band-grid shapes):
    # the default lane replicates all-band stacks; the segment lane shards.
    geom_bytes = (atm_latlon_geometry_bytes(model.grid, nd) if nd > 1
                  else None)

    # Communication accounting (audit item 4) + calibrated T_bound (item 8).
    # nd=1: zero inter-device traffic is a FACT (recorded as 0), so the
    # bound is complete and trivially equals the measured compute.  nd>1:
    # there is no analytic halo-message census for the atm latlon step yet
    # (the ocean twin derives one from its barotropic solver) — the comm
    # ingredients are recorded null with this reason and the bound is
    # emitted incomplete rather than fabricated.
    if nd <= 1:
        _msgs, _bytes_msg, _nred = 0, 0, 0
        _bytes_lower = False   # zero traffic is exact, not an undercount
        _comm_note = "single device: no inter-device halo/reduction traffic"
    else:
        _msgs, _bytes_msg, _nred = None, None, None
        _bytes_lower = None
        _comm_note = ("no analytic halo-message census for the atm latlon "
                      "step yet (audit item 4 follow-up) — comm fields null, "
                      "not fabricated")
    comm_rec = comm_accounting(
        halo_messages_per_step=_msgs,
        bytes_per_message=_bytes_msg,
        full_state_gathers_per_step=0,   # fused scan/segment: no per-step gather
        scope_note=_comm_note,
        bytes_are_lower_bound=_bytes_lower,
    )
    bound_rec = calibrated_bound(
        measured_fused_step_ms=_measured_med,
        # Invalid rows feed the bound NOTHING: even the CLI-provided nd=1
        # baseline is withheld so bound_ingredients.compute_ms cannot dress
        # a diverging row up as a modelled one (codex).
        single_device_fused_step_ms=(
            _measured_med if nd == 1
            else (args.single_dev_fused_ms if valid else None)),
        halo_messages_per_step=comm_rec["halo_messages_per_step"],
        halo_bytes_per_step=comm_rec["halo_bytes_per_step"],
        n_reductions_per_step=_nred,
        # rank imbalance is measured by timed_scan_blocks (default lane);
        # the segment lane records no cross-process block gather -> null.
        rank_imbalance=(float(timing["rank_imbalance"])
                        if timing is not None else None),
        latency_us=args.comm_latency_us,
        bandwidth_GBs=args.comm_bandwidth_gbs,
    )

    rec = dict(
        mode=args.mode, n_devices=nd, n_lat=n_lat, n_lon=args.n_lon,
        nlev=args.nlev, physics=args.physics, steps=args.steps,
        platform=jax.default_backend(),
        n_processes=jax.process_count(),
        multicontroller=bool(args.multicontroller),
        segment_mode=(seg_n > 0),
        segment_steps=(seg_n if seg_n > 0 else None),
        # Measurement validity (codex batch4): finite_ok is the ACCUMULATED
        # in-graph finite verdict (null in the unchecked default fused lane);
        # valid=false marks the row as NOT scaling data; completed_blocks
        # says where a diverging segment run stopped.
        finite_ok=finite_ok,
        valid=valid,
        completed_blocks=completed_blocks,
        steady_median_ms=round(med, 4),
        cells=n_lat * args.n_lon * args.nlev,
    )
    if seg_n > 0:
        # Segment lane: unit 0 = the first BLOCK (includes the scan
        # compile); per-step numbers derive from whole blocks.
total 364
-rw-rw----+ 1 b381103 bd1083 3202 Aug  2 16:52 _probe_latlon_halo_census.py
-rw-rw----+ 1 b381103 bd1083 2979 Aug  2 12:44 mpas_s9_ensemble.sbatch
-rw-rw----+ 1 b381103 bd1083 2318 Jul 31 19:15 mpas_s9_ladder.sbatch
-rw-rw----+ 1 b381103 bd1083 3471 Jul 31 19:15 mpasoc_metis_placement.sbatch
-rw-rw----+ 1 b381103 bd1083 1550 Jul 30 10:50 oc128_bufdump.sbatch
drwxrws---+ 2 b381103 bd1083 4096 Jul 30 10:50 __pycache__
-rw-rw----+ 1 b381103 bd1083 1412 Jul 30 10:50 _bufdump_compileonly.py
-rw-rw----+ 1 b381103 bd1083 1584 Jul 30 02:05 oc128_ab.sbatch
-rw-rw----+ 1 b381103 bd1083  507 Jul 30 01:04 prewarm_s9.sbatch
-rw-rw----+ 1 b381103 bd1083 1849 Jul 29 13:14 fig3_mpas128.sbatch
-rw-rw----+ 1 b381103 bd1083 2324 Jul 29 09:13 fig3_cpu_f32.sbatch
-rw-rw----+ 1 b381103 bd1083 1448 Jul 29 09:13 fig3_oc128.sbatch
-rw-rw----+ 1 b381103 bd1083 1223 Jul 29 09:13 fig3_mpas32.sbatch
-rw-rw----+ 1 b381103 bd1083 1459 Jul 29 09:13 fig3_atm128.sbatch
-rw-rw----+ 1 b381103 bd1083 1568 Jul 28 20:57 memprobe.sbatch
-rw-rw----+ 1 b381103 bd1083 1646 Jul 28 20:46 cube_skew.sbatch
-rw-rw----+ 1 b381103 bd1083 2067 Jul 28 19:54 ll2304_retry.sbatch
-rw-rw----+ 1 b381103 bd1083 4105 Jul 28 18:38 _memprobe_ocean.py
-rw-rw----+ 1 b381103 bd1083 1972 Jul 28 10:04 s9_insurance.sbatch
-rw-rw----+ 1 b381103 bd1083 1665 Jul 28 06:58 gap_tripole16.sbatch
-rw-rw----+ 1 b381103 bd1083 1557 Jul 28 06:57 gap_cube_f64_tiled.sbatch
-rw-rw----+ 1 b381103 bd1083 2165 Jul 28 06:01 cpu1024_matched.sbatch
-rw-rw----+ 1 b381103 bd1083 1615 Jul 28 03:06 ocean_fixedtile32.sbatch
-rw-rw----+ 1 b381103 bd1083 2056 Jul 28 03:05 cube_nsys_fixed.sbatch
-rw-rw----+ 1 b381103 bd1083 2685 Jul 28 02:00 cpu_1024.sbatch
-rw-rw----+ 1 b381103 bd1083 1906 Jul 28 01:30 ocean_scaleout64.sbatch
-rw-rw----+ 1 b381103 bd1083 2400 Jul 27 23:30 mpasoc_partition_2x2.sbatch
-rw-rw----+ 1 b381103 bd1083 1946 Jul 27 23:16 mpasoc_fixed_rpn.sbatch
-rw-rw----+ 1 b381103 bd1083 1705 Jul 27 19:56 mpasoc_scaleout.sbatch
-rw-rw----+ 1 b381103 bd1083 2001 Jul 27 17:47 scaleout_latlon64.sbatch
-rw-rw----+ 1 b381103 bd1083 1878 Jul 27 14:34 cube_fixedtile_small.sbatch
-rw-rw----+ 1 b381103 bd1083 1646 Jul 27 14:13 cube_fixedtile_L30.sbatch
-rw-rw----+ 1 b381103 bd1083 1412 Jul 27 14:12 cpu_s8_np512.sbatch
-rw-rw----+ 1 b381103 bd1083 2153 Jul 27 13:52 scaleout_cube_kt3.sbatch
-rw-rw----+ 1 b381103 bd1083 2010 Jul 27 12:43 scaleout_cube_tiled96.sbatch
-rw-rw----+ 1 b381103 bd1083 2424 Jul 27 12:32 scaleout_cube_matched.sbatch
-rw-rw----+ 1 b381103 bd1083 1839 Jul 27 12:30 scaleout_cpu_big.sbatch
-rw-rw----+ 1 b381103 bd1083 1465 Jul 27 11:53 fig_ico_f64_matched.sbatch
-rw-rw----+ 1 b381103 bd1083 1417 Jul 27 11:50 fig_cube_f64.sbatch
-rw-rw----+ 1 b381103 bd1083 1323 Jul 27 11:22 fig_ico_f32.sbatch
-rw-rw----+ 1 b381103 bd1083 1582 Jul 27 11:09 mpasoc_cpu_ladder.sbatch
-rw-rw----+ 1 b381103 bd1083 1841 Jul 27 11:09 fig_gap_f64.sbatch
-rw-rw----+ 1 b381103 bd1083 1807 Jul 27 10:17 ocean_cov2.sbatch
-rw-rw----+ 1 b381103 bd1083 1520 Jul 27 10:10 mpas_f64_np8_receipt.sbatch
-rw-rw----+ 1 b381103 bd1083 1915 Jul 27 10:04 ocean_coverage_lanes.sbatch
-rw-rw----+ 1 b381103 bd1083 1622 Jul 27 10:03 mpas_f64_ladder.sbatch
-rw-rw----+ 1 b381103 bd1083 2053 Jul 27 10:03 ocean_mixed_precision.sbatch
-rw-rw----+ 1 b381103 bd1083 1311 Jul 26 13:26 wetbal_only.sbatch
-rw-rw----+ 1 b381103 bd1083 2084 Jul 26 13:03 mpas_barrier_ab.sbatch
-rw-rw----+ 1 b381103 bd1083 1886 Jul 26 12:52 mpas_np4_fusion_flags.sbatch
-rw-rw----+ 1 b381103 bd1083 1996 Jul 26 12:46 mpas_np4_hlo.sbatch
-rw-rw----+ 1 b381103 bd1083 1973 Jul 26 12:32 mpas_np4_nsys.sbatch
-rw-rw----+ 1 b381103 bd1083  657 Jul 26 12:26 gather_stencil.sbatch
-rw-rw----+ 1 b381103 bd1083 3067 Jul 26 12:24 cpu_wetbal_numa.sbatch
-rw-rw----+ 1 b381103 bd1083 1908 Jul 25 12:36 ocean_stability_repeats.sbatch
-rw-rw----+ 1 b381103 bd1083 2004 Jul 25 07:12 ocean_widehalo_stability.sbatch
-rw-rw----+ 1 b381103 bd1083 1766 Jul 25 06:40 ocean_crossover_f32_nd4.sbatch
-rw-rw----+ 1 b381103 bd1083 1970 Jul 25 06:24 ocean_config_headtohead.sbatch
-rw-rw----+ 1 b381103 bd1083 1958 Jul 25 05:49 ocean_singlereduce_slope.sbatch
-rw-rw----+ 1 b381103 bd1083 1804 Jul 25 05:16 ocean_iter_compute_check.sbatch
-rw-rw----+ 1 b381103 bd1083 1851 Jul 25 04:41 ocean_pcg_iters_sweep.sbatch
-rw-rw----+ 1 b381103 bd1083 2224 Jul 25 04:04 ocean_overlap_test.sbatch
-rw-r-----+ 1 b381103 bd1083 2757 Jul 25 03:39 probe_sharded_overhead.py
-rw-rw----+ 1 b381103 bd1083  730 Jul 25 03:35 sharded_overhead.sbatch
-rw-r-----+ 1 b381103 bd1083 2394 Jul 25 03:11 probe_ocean_halo_bytes.py
-rw-rw----+ 1 b381103 bd1083 2583 Jul 25 02:34 ocean_roofline_correct.sbatch
-rw-rw----+ 1 b381103 bd1083 1763 Jul 25 02:29 ocean_full_roofline.sbatch
-rw-rw----+ 1 b381103 bd1083 1575 Jul 25 02:24 ocean_calibrated_bound.sbatch
-rw-rw----+ 1 b381103 bd1083 1586 Jul 25 01:25 comm_microbench.sbatch
-rw-rw----+ 1 b381103 bd1083 2127 Jul 24 23:11 mpas_codegen_ab.sbatch
-rw-rw----+ 1 b381103 bd1083 1744 Jul 24 23:01 mpas_partmethod_ab.sbatch
-rw-rw----+ 1 b381103 bd1083  927 Jul 24 22:29 mpas_hlo_diff.sbatch
-rw-rw----+ 1 b381103 bd1083 1842 Jul 24 21:42 mpas_recheck.sbatch
-rw-rw----+ 1 b381103 bd1083 2109 Jul 24 21:32 atm_ladders_final.sbatch
-rw-rw----+ 1 b381103 bd1083 1386 Jul 24 20:37 cube_fair_c768.sbatch
-rw-rw----+ 1 b381103 bd1083 1586 Jul 24 20:11 ocean_weak_prod.sbatch
-rw-rw----+ 1 b381103 bd1083 1386 Jul 24 19:24 cube_fair_c192.sbatch
-rw-rw----+ 1 b381103 bd1083 1386 Jul 24 19:17 cube_fair_ladder.sbatch
-rw-rw----+ 1 b381103 bd1083 1109 Jul 24 18:58 cube_face6_c384.sbatch
-rw-rw----+ 1 b381103 bd1083 1109 Jul 24 18:54 cube_face6_baseline.sbatch
-rw-rw----+ 1 b381103 bd1083 1600 Jul 24 18:50 ocean_ab_vmixf32.sbatch
-rw-rw----+ 1 b381103 bd1083  844 Jul 24 18:40 gate_selfspawn.sbatch
-rw-rw----+ 1 b381103 bd1083 1996 Jul 24 17:53 ocean_ab_xla.sbatch
-rw-rw----+ 1 b381103 bd1083 2017 Jul 24 16:50 anchor_1gpu_derecho_match.sbatch
-rw-rw----+ 1 b381103 bd1083 1404 Jul 24 16:41 ocean_ab_widefused.sbatch
-rw-rw----+ 1 b381103 bd1083 1576 Jul 24 16:40 ocean_ab_ncclproto.sbatch
-rw-rw----+ 1 b381103 bd1083 1758 Jul 24 16:27 ocean_ab_pcg.sbatch
-rw-rw----+ 1 b381103 bd1083 7431 Jul 24 15:25 plot_campaign_cpu_mpi.py
-rw-rw----+ 1 b381103 bd1083 2402 Jul 24 15:15 ocean_ab_solver.sbatch
docs/performance/scaling/derecho_colleague_runbook_2026-07.md
docs/performance/scaling/levante_campaign_2026-07-24.md
docs/performance/scaling/spmd_message_census_2026-07-08.md
scripts/cluster/scaling_derecho/README.md
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch
scripts/plot/plot_scaling_paper_figure.py
scripts/tmp/ocean_ab_xla.sbatch

exec
/usr/bin/bash -lc "sed -n '1,300p' scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch
sed -n '1,300p' scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch
sed -n '1,250p' scripts/tmp/scaleout_latlon64.sbatch
rg -n -C 5 \"def hlo_collective_census|hlo_collective_census\" scripts/bench/metadata.py tests --glob '*.py'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 354ms:
#!/bin/bash -l
#SBATCH --job-name=ll128_comb
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=ll128_comb.%j.log
# CP-COMBINING A/B at LL2048@128 (the measured/bound ~2.9 gap).
# The calibrated bound (census 41 CP + 1 AR/step, exact bytes 18.5 MB/dev,
# IB 26.3us/23.5GB/s, nd=1 same-tile compute 1.659 ms) models 1.85-1.89 ms;
# measured is 5.58 (job 26534060). Leading PLAUSIBLE mechanism: effective
# per-CP overhead (launch+schedule+sync) >> raw fabric latency across 41
# dependency-chained exchanges — the arXiv:2607.16100 regime. Lever:
# COMBINE independent CPs into fewer, larger messages.
# NOTE: the closed-levers null for these flags was the OCEAN lane
# (reduction-dominated); this is the first atm-latlon test — not a rerun
# of a closed null.
# Falsifiability, BEFORE submit — arms byte-matched to 26534060 protocol:
#   A control (default flags)      : expect ~5.6 ms
#   B +cp-combine 8MB threshold    : CONFIRM lever if >=10% under A
#   C +combine +pipelined-p2p      : scheduling interaction
#   REFUTE if B,C within 2% of A -> overhead is not combinable-CP count;
#   next hypothesis = unoverlapped serial chain (scheduling lever).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll128_comb_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
run_arm () { # tag extra_xla_flags
  echo "=== LL2048@128 f32 arm=$1 XLA_EXTRA='$2' ==="
  XLA_FLAGS="${2}" JAX_ENABLE_X64=0 srun --ntasks=128 --ntasks-per-node=4 \
      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices 128 --mode strong \
      --n-lat 2048 --n-lon 4096 --nlev 26 --steps 12 --warmup 3 \
      --out "$OUTDIR/$1.jsonl" || { echo "$1 FAILED"; rc=1; }
}
run_arm A_default ""
run_arm B_combine "--xla_gpu_collective_permute_combine_threshold_bytes=8388608"
run_arm C_combine_pipelined "--xla_gpu_collective_permute_combine_threshold_bytes=8388608 --xla_gpu_enable_pipelined_p2p=true"
echo "=== RESULTS ==="
for T in A_default B_combine C_combine_pipelined; do
  "$PY" -c "
import json,math,sys
try:
    d=json.loads(open('$OUTDIR/$T.jsonl').readline())
    ms=d['steady_median_ms']; assert math.isfinite(ms) and ms>0
except Exception as e:
    print('$T: MISSING/INVALID ->', e); sys.exit(1)
print(f'$T: {ms:8.3f} ms')" || { echo "$T receipt invalid"; rc=1; }
done
echo "DONE rc=$rc"; exit $rc
#!/bin/bash -l
#SBATCH --job-name=atm_ll_192
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=48
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=atm_ll_192.%j.log
# HUNDREDS-OF-GPUS lat-lon atmosphere (user directive 2026-08-02).
# LL2048 does not divide 192 ranks (2048/192 non-integer bands), so the
# >128 strong pair moves to LL2304 (96 and 192 both divide 2304) and the
# above-floor "hundreds" point is LL2880x5760 @192 = 86.4k cols/GPU.
#   arm 1: LL2304x4608 @ 96  f32 (110.6k cols/GPU)
#   arm 2: LL2304x4608 @192  f32 (55.3k cols/GPU — still ABOVE the ~30k
#          floor; codex r21 caught the first draft halving these — so
#          this pair tests device-count cost at healthy tiles)
#   arm 3: LL2880x5760 @192  f32 (86.4k cols/GPU — the honest 192-GPU
#          working point; predicted to hold ~existing GC/s levels)
# Context anchors (same bench, steps 12/warmup 3): LL2048@128 f32
# 5.58 ms 39.11 GC/s (job 26534060).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/atm_ll192_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
run_arm () { # nlat nlon np tag
  echo "=== atm LL$1x$2 @$3 f32 ==="
  JAX_ENABLE_X64=0 srun --ntasks="$3" --ntasks-per-node=4 \
      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices "$3" --mode strong \
      --n-lat "$1" --n-lon "$2" --nlev 26 \
      --steps 12 --warmup 3 \
      --out "$OUTDIR/$4.jsonl" || { echo "$4 FAILED"; rc=1; }
}
run_arm 2304 4608  96 LL2304_f32_np96
run_arm 2304 4608 192 LL2304_f32_np192
run_arm 2880 5760 192 LL2880_f32_np192
echo "=== RESULTS ==="
for T in LL2304_f32_np96 LL2304_f32_np192 LL2880_f32_np192; do
  F="$OUTDIR/$T.jsonl"
  "$PY" -c "
import json,math,sys
try:
    d=json.loads(open('$F').readline())
    ms=d['steady_median_ms']
    assert math.isfinite(ms) and ms > 0
except Exception as e:
    print('$T: MISSING/INVALID ->', e); sys.exit(1)
print(f'$T: {ms:8.2f} ms {d.get(\"mcells_per_s\",0)/1000:6.2f} GC/s')" \
    || { echo "$T receipt invalid"; rc=1; }
done
echo "DONE rc=$rc"; exit $rc
#!/bin/bash -l
#SBATCH --job-name=ll_scaleout
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=ll_2048.%j.log
# Atm lat-lon: more GPUs AND higher resolution, in the matched-pair form
# codex round-14 asked for. The band decomposition has no 6*kt^2-style
# validation ceiling, so this lane can actually reach 64 GPUs.
#
#   LL768x1536  @ 16 = 73.7k cols/GPU   (anchor)
#   LL768x1536  @ 64 = 18.4k            <- fixed problem, 4x N: tile drops
#                                          below the ~30k floor -> should
#                                          show the plateau
#   LL1536x3072 @ 64 = 73.7k            <- SAME TILE as the anchor at 4x N:
#                                          if this holds the anchor time,
#                                          the plateau is tile-floor only
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO"
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll_scaleout_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
run_arm () { # $1 n_lat  $2 ntasks
  echo "=== LL$1x$((2*$1)) @ $2 GPUs ==="
  srun --ntasks="$2" --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none \
       --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices "$2" --mode strong \
      --n-lat "$1" --n-lon $((2*$1)) --nlev 26 \
      --steps 12 --warmup 3 \
      --out "$OUTDIR/LL$1_np$2.jsonl" \
    || { echo "LL$1 @ $2 FAILED"; rc=1; }
}
run_arm 2048 64
echo "=== RESULTS (campaign f32 LL720@16 = 3.54 ms) ==="
for F in "$OUTDIR"/*.jsonl; do
  "$PY" -c "
import json,os
d=json.loads(open('$F').readline())
print(f\"{os.path.basename('$F'):20s} {d['steady_median_ms']:8.2f} ms  prec={d.get('precision')}\")" 2>/dev/null || echo "$(basename $F): no row"
done
echo "DONE rc=$rc"; exit $rc
scripts/bench/metadata.py-200-              for key, rx in _COLLECTIVE_OP_RES.items()}
scripts/bench/metadata.py-201-    counts["total"] = sum(counts.values())
scripts/bench/metadata.py-202-    return counts
scripts/bench/metadata.py-203-
scripts/bench/metadata.py-204-
scripts/bench/metadata.py:205:def hlo_collective_census(fn, *args) -> dict[str, int] | None:
scripts/bench/metadata.py-206-    """Best-effort full collective census of the COMPILED HLO of ``fn(*args)``.
scripts/bench/metadata.py-207-
scripts/bench/metadata.py-208-    Superset of :func:`hlo_collective_permutes` — compiles (so combined /
scripts/bench/metadata.py-209-    pipelined collectives are reflected as executed, not as emitted) and runs
scripts/bench/metadata.py-210-    :func:`count_collectives`.  Returns ``None`` (never raises) if
--
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

exec
/usr/bin/bash -lc "sed -n '130,235p' scripts/bench/metadata.py
sed -n '1,180p' scripts/bench/bench_atm_latlon_spmd_scaling.py
rg -n \"enable_x64|JAX_ENABLE_X64|dtype|float64|x64|precision\" scripts/bench/bench_atm_latlon_spmd_scaling.py packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py scripts/cluster/scaling_levante/atm_l* --glob '*.py' --glob '*.sbatch'
git status --short && git diff -- docs/performance/scaling/levante_campaign_2026-07-24.md scripts/tmp/_probe_latlon_halo_census.py scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 483ms:
    it is the round-count metric #1113 needs to decompose multi-node overhead.
    Matches the op-call form only (StableHLO underscore + optimized-XLA hyphen,
    async ``-start`` counted once, ``-done`` companion excluded by the regex),
    so config-header flag names that merely CONTAIN "collective_permute" never
    inflate the count.  Canonical for every bench that reports
    ``hlo_collective_permutes`` (cube tiled, MPAS ico) — no re-implementation."""
    return _count_op_calls(hlo_text, _COLLECTIVE_PERMUTE_RE)


def hlo_collective_permutes(fn, *args) -> int | None:
    """Best-effort: count the collective-permutes in the COMPILED HLO of
    ``fn(*args)``.

    Compiles (``.lower(...).compile().as_text()``), NOT bare
    ``.lower().as_text()``: the census must reflect the EXECUTABLE's round
    count, because XLA collective-permute combining / pipelined-p2p
    (#1175, ``MPAS_CP_COMBINE``) fuses rounds during optimization — the whole
    metric #1113 tracks. Pre-optimization StableHLO would overstate CPs versus
    the timed executable. Matches the cube tiled bench, which compiles too.
    Returns ``None`` (never raises) if lowering/compilation is unsupported OR
    the backend's ``as_text()`` yields no HLO, so a timing probe can record
    "unknown" rather than crash."""
    import jax
    try:
        text = jax.jit(fn).lower(*args).compile().as_text()
        return count_collective_permutes(text) if text else None
    except Exception:
        return None


def _op_call_re(op_name: str) -> "re.Pattern[str]":
    """Op-call-form matcher for a single HLO collective ``op_name``.

    ``op_name`` is the hyphen spelling (``"all-reduce"``).  Matches BOTH the
    optimized-XLA hyphen and StableHLO underscore spellings, an optional async
    ``-start``/``_start`` suffix, and requires the ``(`` op-call form so a
    config-header ``XLA_FLAGS`` echo that merely CONTAINS the op name can never
    inflate the count (same guard as :data:`_COLLECTIVE_PERMUTE_RE`)."""
    stem = op_name.replace("-", "[_-]")
    return re.compile(stem + r"(?:[_-]start)?\(")


#: Every collective OP family a scaling row can run.  ``collective-permute`` is
#: the band/face halo (reuse the canonical permute regex so its count stays
#: bit-identical to :func:`count_collective_permutes`); ``all-reduce`` is the
#: conservation fixer AND the ocean implicit-CN PCG reduction wall (~120/step —
#: the #1 ocean strong-scaling bottleneck, invisible to a permute-only census);
#: the rest surface any SPMD resharding an operator introduces.
_COLLECTIVE_OP_RES: dict[str, "re.Pattern[str]"] = {
    "collective_permute": _COLLECTIVE_PERMUTE_RE,
    "all_reduce": _op_call_re("all-reduce"),
    "all_gather": _op_call_re("all-gather"),
    "all_to_all": _op_call_re("all-to-all"),
    "reduce_scatter": _op_call_re("reduce-scatter"),
}


def count_collectives(hlo_text: str) -> dict[str, int]:
    """Full per-family collective census of a lowered/compiled HLO text dump.

    Superset of :func:`count_collective_permutes`: adds ``all_reduce`` (the
    reduction wall that dominates ocean implicit-CN strong scaling and hides
    from a permute-only count), ``all_gather``, ``all_to_all``,
    ``reduce_scatter``.  Same STATIC, op-call-form discipline (see
    :func:`_count_op_calls`: config-header flag names never inflate; an async
    collective counts once via ``-start`` with its ``-done`` companion excluded
    by the regex, NOT by a fragile line-wide substring test).  Keys are
    underscore-normalized op names plus a ``total``.  Canonical census for the
    scaling-diagnosis tool and the SPMD benches — no re-implementation."""
    counts = {key: _count_op_calls(hlo_text, rx)
              for key, rx in _COLLECTIVE_OP_RES.items()}
    counts["total"] = sum(counts.values())
    return counts


def hlo_collective_census(fn, *args) -> dict[str, int] | None:
    """Best-effort full collective census of the COMPILED HLO of ``fn(*args)``.

    Superset of :func:`hlo_collective_permutes` — compiles (so combined /
    pipelined collectives are reflected as executed, not as emitted) and runs
    :func:`count_collectives`.  Returns ``None`` (never raises) if
    lowering/compilation is unsupported OR the backend's ``as_text()`` yields
    no HLO, so a probe records "unknown" not a crash.

    NOTE the count is device-count- and FLAG-dependent: the DEFAULT schedule is
    reproducible from a CPU virtual-device compile, but GPU-only XLA collective
    combining / pipelined-p2p (``--xla_gpu_collective_permute_combine_*``, lane
    T) can lower the optimized count — which is exactly why the cluster jobs run
    this against the REAL on-device executable, not a CPU proxy."""
    import jax
    try:
        text = jax.jit(fn).lower(*args).compile().as_text()
        return count_collectives(text) if text else None
    except Exception:
        return None


def _is_empty(v: Any) -> bool:
    """True for a non-informative required value: ``None``, ``""``, or an EMPTY
    container (e.g. ``precision_knobs={}`` — which would hide an f32/TF32
    ablation).  Scalars ``0`` / ``0.0`` / ``False`` are NOT empty (a legitimate
    ``n_gpus=0`` / ``gpu_direct_active=False`` must pass)."""
    if v is None or v == "":
        return True
    if isinstance(v, (dict, list, tuple, set, frozenset)) and len(v) == 0:
        return True
"""Strong/weak scaling bench for the lat-band SPMD lat-lon C-grid hydrostatic
atm step (make_sharded_atm_latlon_step / run_atm_latlon_spmd, the A1 work).

Times the SHARDED step across an N-device ("lat",) mesh and reports per-step
wall time + speedup vs 1 device.  The DEFAULT lane follows the M1 measurement
contract (``metadata.timed_scan_blocks``): fused ``lax.scan`` blocks of
``--steps`` steps with device sync only AROUND each block (``fused_step_ms``,
slowest process across controllers) plus a SEPARATE individually-synced
dispatch-latency probe (``step_latency_ms``) — never mixed.  The
jit(shard_map) step is built once and cached by make_sharded_atm_latlon_step;
re-tracing would show up as every block paying the scan-compile cost again.

  strong: fixed (n_lat, n_lon, nlev), vary n_devices -> speedup = t(1)/t(n).
  weak:   n_lat = nlat_per_dev * n_devices (fixed per-device rows) -> ideal flat.

``--segment-steps N`` (M2b): times the COMPILED-SEGMENT lane instead — ONE
jitted lax.scan of N sharded steps per block (make_sharded_atm_latlon_segment,
band-SHARDED geometry, in-graph finite scalar), so --steps counts BLOCKS of N
steps and the per-step numbers derive from whole-block wall times.  Unlike the
default lane (which scans the bench-local step fn), the segment is the
PRODUCTION artifact; the record carries ``segment_mode=true`` +
``per_block_ms`` and the per-device geometry bytes (replicated vs
band-sharded) computed from the real band-grid shapes.

Receipt honesty: a segment whose in-graph finite scalar reports a non-finite
state STOPS the timed loop and stamps the record ``finite_ok=false`` +
``valid=false`` (with ``completed_blocks`` saying how far it got, a
``diverged: ...`` entry under ``metadata._incomplete``, and nulled
``sypd``/``mcells_per_s``) — a diverging trajectory is never serialized as
valid scaling data.  The default fused lane has no in-graph finite check, so
its rows carry ``finite_ok=null``.

Device count is fixed at process start, so each n_devices runs as a SEPARATE
process (one sbatch step per count); this script benches ONE n_devices and
appends a JSON line. JAX_PLATFORMS=cpu with --xla_force_host_platform_device_count
gives virtual CPU devices (communication-overhead characterization, NOT a real
speedup); a real number needs one GPU per band.

Multi-controller (route-B, ``--multicontroller``): the lat-lon analogue of the
cubed-sphere ``run_cpu_mpi_scaling --cs-spmd`` A1 path. Every process calls
``jax.distributed.initialize`` BEFORE any other JAX use, the ("lat",) mesh is
built over the GLOBAL ``jax.devices()`` (all processes), and the existing
``make_sharded_atm_latlon_step`` + band-ppermute halo runs unchanged — the
ppermute/psum collectives cross processes via the distributed runtime (NCCL on
GPU / gloo on CPU). NO mpi4jax is armed in this mode (mixing the mpi4jax halo
machinery with jax.distributed collectives in one program is the documented
mixed-stack deadlock hazard — see run_cpu_mpi_scaling._build_cubed_sphere_spmd).
This is the halo path that keeps intra-node GPU traffic on NCCL and bypasses
the host-staged / CXI-inject-broken cross-node GPU-direct MPI route (see
docs/performance/multinode_gpu_direct_cxi.md).

Launch (cluster, one process per GPU):
  srun -n 8 python bench_atm_latlon_spmd_scaling.py --multicontroller \
      --n-devices 8 ...            # SLURM: coordinator auto-detected
  mpiexec -n 8 python ... --multicontroller --coordinator host0:9876
"""
from __future__ import annotations

import argparse
import json
import os
import time

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Bench dir for the shared metadata module (sibling-script import pattern —
# needed when this file is loaded by path from tests, not run as a script).
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Shared self-describing scaling metadata (anti-fake-scaling audit): merged
# under rec["metadata"] so a virtual-CPU-device proxy, a gloo/TCP fabric run,
# or an f32 ablation is falsifiable from the JSONL row alone.  metadata.py
# imports JAX lazily, so this is safe before jax.distributed.initialize.
from metadata import (  # noqa: E402
    annotate_incomplete,
    calibrated_bound,
    comm_accounting,
    scaling_metadata,
    tidy_throughput_fields,
)


def _build_model(n_lat, n_lon, nlev):
    from legoesm import constants
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig, CGridLatLonPrimitiveEquationModel)
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon, radius=constants.R_earth,
                              omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=nlev)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True, use_polar_filter=False, use_ppm_transport=True,
        time_integrator="ssp_rk3")
    return CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)


def _build(n_lat, n_lon, nlev):
    # nd=1 lane + tests: global (unsharded) IC build, unchanged protocol.
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_latlon
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        hydrostatic_to_cgrid)

    model = _build_model(n_lat, n_lon, nlev)
    hs0 = held_suarez_init_latlon(model.grid, model.sigma_coord)
    c0 = hydrostatic_to_cgrid(hs0, model.grid)
    return model, c0


def _block(state):
    jax.block_until_ready(jax.tree.leaves(state))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-lat", type=int, default=128)
    p.add_argument("--n-lon", type=int, default=256)
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--n-devices", type=int, required=True)
    p.add_argument("--mode", choices=["strong", "weak"], default="strong")
    p.add_argument("--nlat-per-dev", type=int, default=32,
                   help="weak mode: lat rows per device")
    p.add_argument("--steps", type=int, default=12,
                   help="Default lane: steps per fused lax.scan timing "
                        "block. Segment mode: number of timed BLOCKS of "
                        "--segment-steps steps each.")
    p.add_argument("--warmup", type=int, default=2,
                   help="Segment mode: timed blocks dropped from steady "
                        "stats (block 0 includes the scan compile). Default "
                        "lane: retained for CLI compat (fused-block timing "
                        "separates compile/probe/blocks explicitly).")
    p.add_argument("--blocks", type=int, default=2,
                   help="Default lane: timed fused blocks (per-block times "
                        "expose drift).")
    p.add_argument("--probe-steps", type=int, default=3,
                   help="Default lane: individually-synced steps for the "
                        "SEPARATE dispatch-latency probe (step_latency_ms).")
    p.add_argument("--segment-steps", type=int, default=0,
                   help="M2b: >0 compiles ONE lax.scan segment of this many "
                        "steps (built once, reused; band-sharded geometry) "
                        "and times BLOCKS of segment calls instead of "
                        "per-step host dispatch. 0 = default fused lane.")
    p.add_argument("--physics", choices=["none", "held_suarez"], default="none")
    p.add_argument("--dt", type=float, default=60.0)
    p.add_argument("--single-dev-fused-ms", type=float, default=None,
                   help="fused_step_ms of the nd=1 row at the SAME per-device "
                        "size (compute ingredient of the calibrated T_bound, "
                        "audit item 8). Omitted at nd>1 -> bound emitted null "
                        "+ flagged incomplete; nd=1 uses its own measurement.")
    p.add_argument("--comm-latency-us", type=float, default=None,
                   help="MEASURED per-message latency [us] of THIS machine's "
                        "fabric. Default: MACHINE-CALIBRATED-REQUIRED "
                        "placeholder in metadata.py -> bound_calibrated=false.")
    p.add_argument("--comm-bandwidth-gbs", type=float, default=None,
                   help="MEASURED link bandwidth [GB/s] of THIS machine's "
                        "fabric. Default: MACHINE-CALIBRATED-REQUIRED "
                        "placeholder in metadata.py -> bound_calibrated=false.")
    p.add_argument("--out", type=str, default="results/a1/spmd_scaling.jsonl")
    p.add_argument("--multicontroller", action="store_true",
                   help="Route-B multi-controller: jax.distributed.initialize "
                        "per process, ('lat',) mesh over the GLOBAL device set "
                        "(one process per GPU / per CPU-device group). NO "
                        "mpi4jax. --n-devices must equal the global device "
                        "count.")
    p.add_argument("--coordinator", type=str, default=None,
                   help="host:port for jax.distributed when auto-detection "
                        "(SLURM) is unavailable; process count/id then come "
                        "from OMPI_COMM_WORLD_SIZE/RANK.")
    args = p.parse_args()

    # Validate the schedule BEFORE any model/device work: a zero/negative
    # --steps would otherwise surface only as timed_scan_blocks' None
    # headline (default lane) or an empty timed loop (segment mode) after
    # the expensive build (the ocean twin's guard).
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch:28:run_arm () { # nlat x64 tag
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch:29:  echo "=== nd=1 LL$1x4096 x64=$2 ($3) ==="
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch:30:  JAX_ENABLE_X64=$2 srun --ntasks=1 --gpus=1 --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:40:  XLA_FLAGS="${2}" JAX_ENABLE_X64=0 srun --ntasks=128 --ntasks-per-node=4 \
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:36:  JAX_ENABLE_X64=0 srun --ntasks="$3" --ntasks-per-node=4 \
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:27:  JAX_ENABLE_X64=1 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=32 \
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:31:      --precision float64 --n-levels 26 --resolution 512 \
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:123:    from legoesm.core.precision import get_policy
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:125:    _dtype = get_policy().storage
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:134:    pert2d = jax.random.normal(key, (n_lat, n_lon), dtype=_dtype) * jnp.asarray(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:135:        perturbation_amplitude, dtype=_dtype
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:146:        return lambda idx: jnp.zeros(_slice_shape(gshape, idx), dtype=_dtype)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:153:        block = jnp.ones(shape, dtype=_dtype) * T_init
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:159:        phis_block = jnp.zeros(shape, dtype=_dtype)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:161:            -phis_block / (constants.R_d * T_init))).astype(_dtype)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:313:    bad = jnp.zeros((), dtype=jnp.int32)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:438:def _dtype_sig(tree) -> tuple:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:440:    every array leaf's ``(shape, dtype, weak_type)``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:443:    dtype and weak type — dtype strings alone would mark a step that flips
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:444:    weak typing (or shape) while preserving dtypes as "stable" and then
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:450:            tuple((tuple(leaf.shape), str(leaf.dtype),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:452:                  for leaf in leaves if hasattr(leaf, "dtype")))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:455:def unroll_to_dtype_fixed_point(step1, state, n_left: int):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:456:    """Unroll ``step1`` applications until the state's dtype signature is a
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:462:    ``lax.scan``-of-a-step needs carry-in == carry-out dtypes, and this is
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:465:    Why: ``lax.scan`` needs carry-in == carry-out dtypes.  A mixed-precision
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:467:    x64 — the f64 sigma-coordinate arrays and the strong-f64
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:476:    ``jax.eval_shape`` probes the step's output dtypes ABSTRACTLY at trace
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:478:    NONE for an already dtype-stable state (the parity-gate f64 states scan
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:483:    strictly monotone leaf promotion over a finite dtype lattice guarantees
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:488:           and _dtype_sig(jax.eval_shape(step1, state)) != _dtype_sig(state)):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:643:    # recompiles the (large, un-jitted) band step EVERY call — a 32x64x10 nd=2
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:733:    Carry dtype: leading steps are UNROLLED outside the ``lax.scan`` until
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:734:    the state's dtype signature is a fixed point of the step
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:735:    (:func:`unroll_to_dtype_fixed_point` — ``jax.eval_shape`` probe, zero
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:736:    FLOPs, trace-time constant).  A mixed-precision IC promotes over the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:741:    per-step lane — never a silent precision change.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:767:            # Unroll to the scan-carry dtype fixed point (helper docstring).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:768:            out, n_left = unroll_to_dtype_fixed_point(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:799:        # Unroll to the scan-carry dtype fixed point (helper docstring).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:800:        out, n_left = unroll_to_dtype_fixed_point(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1382:    axes, leading steps unrolled to the scan-carry dtype fixed point via
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1383:    :func:`unroll_to_dtype_fixed_point`, STATELESS physics only,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1421:        # Unroll to the scan-carry dtype fixed point (helper docstring).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1425:        out, n_left = unroll_to_dtype_fixed_point(
scripts/bench/bench_atm_latlon_spmd_scaling.py:404:        precision="float64" if jax.config.jax_enable_x64 else "float32",
scripts/bench/bench_atm_latlon_spmd_scaling.py:432:        precision="float64" if jax.config.jax_enable_x64 else "float32",
 M docs/performance/scaling/levante_campaign_2026-07-24.md
?? .physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md
?? scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch
?? scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch
?? scripts/cluster/scaling_levante/mpas_s10_128.sbatch
?? scripts/cluster/scaling_levante/mpas_s10_192.sbatch
diff --git a/docs/performance/scaling/levante_campaign_2026-07-24.md b/docs/performance/scaling/levante_campaign_2026-07-24.md
index ef52ca1b7..dbd844ec7 100644
--- a/docs/performance/scaling/levante_campaign_2026-07-24.md
+++ b/docs/performance/scaling/levante_campaign_2026-07-24.md
@@ -1791,3 +1791,44 @@ meshes pad +126 cells; tiles NEAR-matched to 0.015 %:
   partition-cut explanations, on the CPU lane at least).
 * The non-monotone tile dependence of the ratio (largest at the
   LARGEST tile, 1.90 at 81.9k) is unexplained; recorded, not theorised.
+
+## Distance-to-modeled-limit: atm lat-lon GPU (2026-08-02, "near theoretical limit" directive)
+
+Closed the bench's own honest-null bound gap (audit item 4) for the
+lat-lon lane, using only repo instruments:
+
+* **Halo census** (new probe `scripts/tmp/_probe_latlon_halo_census.py`,
+  virtual-CPU forced-host-platform lowering of the REAL
+  `make_sharded_atm_latlon_step`): **41 collective-permutes + 1
+  all-reduce per step**, nd-INDEPENDENT (identical at nd=8 and nd=16 —
+  the 1-D band structure check). Exact CP payload from compiled-HLO
+  result shapes: 4,635,408 B/dev/step at n_lon=1024 L26 f32 = 1.06x the
+  single-row slab model; linear in n_lon (checked 1024 vs 2048, 0.07 %
+  residual) -> **18.5 MB/dev/step at n_lon=4096 f32**. CAVEAT: CPU
+  lowering; GPU-side collective combining could change the executed
+  count (metadata.py:214) — the bound is a MODEL.
+* **Same-tile nd=1 compute baselines** (job 26630370, roofline recipe):
+  16x4096 f32 1.659 ms, 32x4096 f32 2.837, 16x4096 f64 2.973.
+  Approximation, recorded: nd=1 includes pole tiles -> compute term
+  biased HIGH -> bound conservative.
+* **Calibrated bound** (`metadata.calibrated_bound`, measured fabric
+  constants: IB 26.3 us / 23.5 GB/s, NVLink 17.8 / 64.2):
+
+| row | measured | t_bound (IB) | measured/bound |
+|---|---|---|---|
+| LL2048@64 f32 | 6.732 | 2.863 | **2.35** |
+| LL2048@128 f32 | 5.577 | 1.848 | **3.02** |
+| LL2048@128 f64 | 9.602 | 2.999 | **3.20** |
+
+* **The lat-lon GPU panel sits ~2.4-3.2x ABOVE its modeled limit** — the
+  eff-0.60 strong leg is NOT close to the fabric+compute floor. Leading
+  PLAUSIBLE mechanism (uninstrumented): effective per-CP cost
+  (launch + schedule + stream sync) well above the raw 26 us fabric
+  latency across 41 dependency-chained exchanges — the arXiv:2607.16100
+  small-collective regime. The model itself notes the serialized-latency
+  vs overlap biases pull opposite ways; treat measured/bound as a
+  consistency diagnostic, not proven headroom.
+* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
+  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
+  same-job control, falsifiability block in the script. The ocean-lane
+  null for these flags was reduction-dominated — first atm test.

exec
/usr/bin/bash -lc "sed -n '1,260p' .physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md 2>/dev/null || true
sed -n '1,240p' scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch
sed -n '100,190p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py
sed -n '430,510p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py
sed -n '380,445p' scripts/bench/bench_atm_latlon_spmd_scaling.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 231ms:
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
session id: 019fc2f8-a371-7da0-8c7a-70a10de177bf
--------
user
Round-7 review: the new distance-to-modeled-limit analysis (last section of docs/performance/scaling/levante_campaign_2026-07-24.md) + instruments. Attack:
1. Census probe scripts/tmp/_probe_latlon_halo_census.py: virtual-CPU forced-host lowering of make_sharded_atm_latlon_step, counted 41 collective-permute + 1 all-reduce per step, identical at nd=8/16; exact CP result bytes summed from compiled HLO text via regex on 'collective-permute(' lines (4,635,408 B at n_lon=1024 L26 f32; linear in n_lon 0.07% residual). Is the regex robust (start/done pairs double-count? async pairs? result-shape vs operand-shape)? Is nd-independence at 8/16 sufficient to claim it for 128?
2. Bound arithmetic: calibrated_bound with compute=nd1 same-tile receipts (job 26630370: 1.659/2.837/2.973 ms), msgs=41, bytes=18.5MB (f32@4096) scaled x2 for f64, nred=1, imbalance from rows, IB 26.3us/23.5GB/s: t_bound 2.863/1.848/2.999 -> measured/bound 2.35/3.02/3.20 for LL2048@64 f32 / @128 f32 / @128 f64. Recheck numbers (measured 6.7324/5.5767/9.6015).
3. Claims wording: '~2.4-3.2x above modeled limit', mechanism labelled PLAUSIBLE (per-CP effective overhead, arXiv:2607.16100 regime), A/B lever job 26630438 with falsifiability block (CONFIRM >=10%, REFUTE <=2%). Any overclaim/confound (e.g. f64 bytes: do 2-D f32 fields stay f32 under x64? census was f32-only), pole-tile compute bias direction, the ocean-null distinction argument.
End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + numbered issues.
codex
I’ll audit the cited section and probe script against the available artifacts, then recalculate the bound from the stated inputs and flag any claims that need narrowing.
exec
/usr/bin/bash -lc "pwd && rg --files -g 'levante_campaign_2026-07-24.md' -g '_probe_latlon_halo_census.py' -g '26630370*' -g '26630438*' -g '*.csv' -g '*.json' | sed -n '1,240p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 434ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.zenodo.json
tests/visual_baselines/sw_cube_vwind.json
docs/performance/scaling/barotropic_mcut.csv
docs/performance/scaling/barotropic_mcut_np4.csv
docs/performance/scaling/barotropic_mcut_np2.csv
docs/performance/scaling/scaling_indicators.csv
docs/performance/scaling/levante_campaign_2026-07-24.md
tests/ocean/fidelity/fixtures/tier8_global_realistic.json
tests/ocean/fidelity/fixtures/tier5_baroclinic_instability.json
tests/ocean/fidelity/fixtures/tier0_invariants.json
tests/ocean/fidelity/fixtures/tier6_channel_circulation.json
tests/ocean/fidelity/fixtures/tier1_linear_waves.json
tests/ocean/fidelity/fixtures/tier7_dino.json
tests/ocean/fidelity/fixtures/tier4_wind_driven_gyres.json
tests/ocean/fidelity/fixtures/tier3_process_benchmarks.json
tests/ocean/fidelity/fixtures/tier2_geostrophic_thermalwind.json
tests/unit/baselines/aerobulk_noskin_v1.json
tests/validation/scm_oracle/manifest.json
tests/unit/golden/yaml_to_experiment/williamson_test2.json
tests/unit/golden/yaml_to_experiment/williamson_test5.json
config/wb/sota/wb2_headline_rmse.csv
config/wb/sweep/stage1/manifest.json
scripts/cluster/ec_site/ec_site_soil_texture.csv
config/aimip/t106_headtohead/winner.json
scripts/validate/ec_site_regression_lai.csv
config/aimip/sweep/stage2/stage1_winners.json
config/aimip/sweep/stage2/manifest.json
scripts/validate/ec_site_regression_golden.json
config/aimip/sweep/stage1/manifest.json

exec
/usr/bin/bash -lc "rg -n -C 5 \"modeled limit|calibrated_bound|26630370|26630438|distance-to|collective-permute|ocean-null|PL[A-Z]+IBLE|CONFIRM|REFUTE\" docs/performance/scaling/levante_campaign_2026-07-24.md scripts tmp . --glob '"'!*.ipynb'"' 2>/dev/null" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 2 in 4207ms:
docs/performance/scaling/levante_campaign_2026-07-24.md-100-OPEN ANOMALY, characterised not explained: per-GPU throughput dips at
docs/performance/scaling/levante_campaign_2026-07-24.md-101-np=4 (247 Mc/s/GPU vs 430 at np2 and 306 at np8), so np4 is barely faster
docs/performance/scaling/levante_campaign_2026-07-24.md-102-than np2 while np8 is 2.5x faster than np4. Evidence gathered:
docs/performance/scaling/levante_campaign_2026-07-24.md-103-- REPRODUCIBLE: two repeats per arm agree within 1 % (19.92/19.87,
docs/performance/scaling/levante_campaign_2026-07-24.md-104-  17.14/17.04, 6.87/6.97).
docs/performance/scaling/levante_campaign_2026-07-24.md:105:- PLACEMENT REFUTED: np4 packed on one node (17.09 ms) == np4 spread over
docs/performance/scaling/levante_campaign_2026-07-24.md-106-  two nodes (17.12 ms), so node crossing is irrelevant.
docs/performance/scaling/levante_campaign_2026-07-24.md:107:- HALO VOLUME REFUTED: ghost-cell census on the padded mesh gives
docs/performance/scaling/levante_campaign_2026-07-24.md-108-  1540/1587/1400/1136 ghost cells per device at np 2/4/8/16 — flat to
docs/performance/scaling/levante_campaign_2026-07-24.md-109-  falling, and under 3 % of owned cells at every count.
docs/performance/scaling/levante_campaign_2026-07-24.md:110:- COLLECTIVE COUNT REFUTED (HLO census, ico L7, CPU virtual devices —
docs/performance/scaling/levante_campaign_2026-07-24.md-111-  device count is a compile-time property so the HLO matches what the GPUs
docs/performance/scaling/levante_campaign_2026-07-24.md:112:  execute): collective-permutes per step are 3 / 9 / 21 at np 2/4/8, i.e.
docs/performance/scaling/levante_campaign_2026-07-24.md-113-  np8 issues 2.3x MORE collectives than np4 and still runs 2.5x faster.
docs/performance/scaling/levante_campaign_2026-07-24.md-114-  Collective COUNT therefore cannot explain the np4 dip (this assumes cost
docs/performance/scaling/levante_campaign_2026-07-24.md-115-rises with count; a per-message-size effect is not excluded). (Fusion count 136/173/240,
docs/performance/scaling/levante_campaign_2026-07-24.md-116-  bitcasts 526/582/694 — the np8 program is finer-grained.)
docs/performance/scaling/levante_campaign_2026-07-24.md:117:- PARTITION METHOD REFUTED (job 26455829): the dip is method-independent —
docs/performance/scaling/levante_campaign_2026-07-24.md-118-  np4/np8 = 17.22/7.00 ms (sfc), 17.20/6.97 (metis), 19.35/6.26
docs/performance/scaling/levante_campaign_2026-07-24.md-119-  (geometric). Every method shows the same 2.5-3.1x jump.
docs/performance/scaling/levante_campaign_2026-07-24.md:120:- XLA CODEGEN ENV KNOBS REFUTED (job 26455948): np4 is 17.12 ms base,
docs/performance/scaling/levante_campaign_2026-07-24.md-121-  17.06 autotune-level-4, 17.10 latency-hiding-off, 17.01
docs/performance/scaling/levante_campaign_2026-07-24.md-122-  command-buffers-off — every arm within 1 %, none recovers np4.
docs/performance/scaling/levante_campaign_2026-07-24.md-123-  (The multi-output-fusion arm errored on an unsupported flag name and is
docs/performance/scaling/levante_campaign_2026-07-24.md-124-  not counted.)
docs/performance/scaling/levante_campaign_2026-07-24.md-125-VERDICT: five hypotheses refuted by measurement (placement, halo volume,
--
docs/performance/scaling/levante_campaign_2026-07-24.md-132-PRACTICAL GUIDANCE MEANWHILE: run this grid at np>=8, where per-device
docs/performance/scaling/levante_campaign_2026-07-24.md-133-throughput is 304-340 Mc/s/GPU vs 220-248 at np4.
docs/performance/scaling/levante_campaign_2026-07-24.md-134-
docs/performance/scaling/levante_campaign_2026-07-24.md-135-RESOLVED 2026-07-26 (nsys job 26479922 + HLO dump 26480096 + sqlite
docs/performance/scaling/levante_campaign_2026-07-24.md-136-timeline): the dip is an XLA CODEGEN pathology, localized to named
docs/performance/scaling/levante_campaign_2026-07-24.md:137:kernels. CONFIRMED: (1) the dip reproduces under nsys with matched
docs/performance/scaling/levante_campaign_2026-07-24.md-138-protocol (L8, padded-16 mesh: 21.01/17.78/7.17 ms at np2/4/8 vs campaign
docs/performance/scaling/levante_campaign_2026-07-24.md-139-19.90/17.09/6.92 — ~5% profiler overhead); (2) at np4 ONLY, giant
docs/performance/scaling/levante_campaign_2026-07-24.md-140-serialized "loop fusion" kernels appear — loop_add_fusion_1/2 at 3.6 ms
docs/performance/scaling/levante_campaign_2026-07-24.md-141-per launch (vs ~3 us for ordinary elementwise kernels) plus a THIRD
docs/performance/scaling/levante_campaign_2026-07-24.md-142-once-per-step group (the unsuffixed loop_add_fusion: 12 of its 44
--
docs/performance/scaling/levante_campaign_2026-07-24.md-146-wait): 3.6 + 3.6 + 3.3 ~= 10.5-10.9 ms/step = the np4 excess; (3) in the optimized
docs/performance/scaling/levante_campaign_2026-07-24.md-147-step HLO these are mega-fusions ON THE HALO PATH: `%loop_add_fusion =
docs/performance/scaling/levante_campaign_2026-07-24.md-148-f32[491520,26]` (edge-tendency add chain, 22 operands incl. an
docs/performance/scaling/levante_campaign_2026-07-24.md-149-input_scatter_fusion) and `%loop_add_fusion.4 = f32[163844,26]` (cell
docs/performance/scaling/levante_campaign_2026-07-24.md-150-array), with the shard_map halo-pack concatenates taking the same adds +
docs/performance/scaling/levante_campaign_2026-07-24.md:151:parameter lists as operands. PLAUSIBLE (inferred from kInput fusion
docs/performance/scaling/levante_campaign_2026-07-24.md-152-semantics + operand lists, not separately timed): the emitter RECOMPUTES
docs/performance/scaling/levante_campaign_2026-07-24.md-153-the expensive scatter+add chain inside each consumer fusion, which is why
docs/performance/scaling/levante_campaign_2026-07-24.md-154-the cost multiplies. WHY np4: fusion cost-model decisions depend on the
docs/performance/scaling/levante_campaign_2026-07-24.md-155-shard shape; at np2/np8 the mega-fusion is not built. This also explains
docs/performance/scaling/levante_campaign_2026-07-24.md-156-why the earlier env-knob sweep missed it — autotune/latency-hiding flags
--
docs/performance/scaling/levante_campaign_2026-07-24.md-211-multinode improvement measured at LL576 was partly a floor effect, and at
docs/performance/scaling/levante_campaign_2026-07-24.md-212-a production tile the identical code scales substantially better (0.37 ->
docs/performance/scaling/levante_campaign_2026-07-24.md-213-0.63). Per-device throughput also rises (259 -> 305 Mc/s/GPU at np4).
docs/performance/scaling/levante_campaign_2026-07-24.md-214-Config is byte-identical between the two rows; only the grid changes.
docs/performance/scaling/levante_campaign_2026-07-24.md-215-
docs/performance/scaling/levante_campaign_2026-07-24.md:216:REFUTED EN ROUTE: the np8 leg timed out twice (>90 min still tracing) while
docs/performance/scaling/levante_campaign_2026-07-24.md-217-np4 — a LARGER per-device tile — finished in ~25 min, which looked like a
docs/performance/scaling/levante_campaign_2026-07-24.md-218-compile-time cliff at that device count. It is not: the third attempt ran
docs/performance/scaling/levante_campaign_2026-07-24.md-219-the identical configuration in **99 seconds** with a 21.6 s compile (job
docs/performance/scaling/levante_campaign_2026-07-24.md-220-26457693). The earlier hangs were transient/environmental, not
docs/performance/scaling/levante_campaign_2026-07-24.md-221-reproducible, and no compile-time defect is claimed.
--
docs/performance/scaling/levante_campaign_2026-07-24.md-246-  differ in machine (A100-80 SXM vs A100-40), jax/tree version and date,
docs/performance/scaling/levante_campaign_2026-07-24.md-247-  so the ~13% gap is not attributable to any single factor.
docs/performance/scaling/levante_campaign_2026-07-24.md-248-- Cube "404 vs 141 Mc/s": the 404 is the single-GPU RTX-5090 Held-Suarez
docs/performance/scaling/levante_campaign_2026-07-24.md-249-  row in `SCALING_SUMMARY.md` SS1; ours is gray+SBM on A100 (job 26445836).
docs/performance/scaling/levante_campaign_2026-07-24.md-250-  That file's own tier table prices gray+SBM ~3x Held-Suarez, so ~135 is
docs/performance/scaling/levante_campaign_2026-07-24.md:251:  the expected equivalent vs 141 measured. PLAUSIBLE reconciliation from
docs/performance/scaling/levante_campaign_2026-07-24.md-252-  two published tables, NOT a matched A/B (GPU, physics and date differ).
docs/performance/scaling/levante_campaign_2026-07-24.md-253-- Ocean absolutes (A100 f64 165-201 Mc/s, f32 352, job 26445836; 5090 f64
docs/performance/scaling/levante_campaign_2026-07-24.md-254-  152 / f32 400 from `SCALING_SUMMARY.md` SS1) are of the same order -
docs/performance/scaling/levante_campaign_2026-07-24.md-255-  again a cross-machine sanity check, not a controlled comparison.
docs/performance/scaling/levante_campaign_2026-07-24.md-256-- Ginsburg "0.92 eff @2 GPU" reconciled: the old bench silently defaulted
--
docs/performance/scaling/levante_campaign_2026-07-24.md-350-Bound = per-device compute (32.58 / 14.63 ms) + modelled comm (2.21) +
docs/performance/scaling/levante_campaign_2026-07-24.md-351-modelled reduction (2.19). The unmodelled gap is **~5 ms/step and roughly
docs/performance/scaling/levante_campaign_2026-07-24.md-352-FLAT** with device count (7.9 ms at nd2, 6.7 at nd4), which is why the
docs/performance/scaling/levante_campaign_2026-07-24.md-353-ratio worsens as compute shrinks.
docs/performance/scaling/levante_campaign_2026-07-24.md-354-
docs/performance/scaling/levante_campaign_2026-07-24.md:355:WHAT THE GAP IS NOT — the omitted-traffic explanation is REFUTED
docs/performance/scaling/levante_campaign_2026-07-24.md-356-(`scripts/tmp/probe_ocean_halo_bytes.py`, HLO byte census on CPU virtual
docs/performance/scaling/levante_campaign_2026-07-24.md-357-devices). The bench's `comm_scope_note` correctly warns that its census is
docs/performance/scaling/levante_campaign_2026-07-24.md-358-"barotropic implicit-CN PCG scope only … baroclinic 3-D pads NOT counted",
docs/performance/scaling/levante_campaign_2026-07-24.md-359-and the true volume IS much larger: **16.22 MB/step across 110
docs/performance/scaling/levante_campaign_2026-07-24.md:360:collective-permutes vs the censused 2.25 MB — a 7.2x undercount**. But
docs/performance/scaling/levante_campaign_2026-07-24.md-361-completing the census moves the bound by only **0.22 ms**, because the
docs/performance/scaling/levante_campaign_2026-07-24.md-362-comm term is LATENCY-dominated: at 122 messages x 17.82 us the latency part
docs/performance/scaling/levante_campaign_2026-07-24.md-363-is 2.174 ms while even 16 MB at 64.22 GB/s is just 0.253 ms.
docs/performance/scaling/levante_campaign_2026-07-24.md-364-
docs/performance/scaling/levante_campaign_2026-07-24.md-365-So with the byte census completed the unexplained residual is still 5.5 ms
docs/performance/scaling/levante_campaign_2026-07-24.md-366-(nd2) and 4.3 ms (nd4).
docs/performance/scaling/levante_campaign_2026-07-24.md-367-
docs/performance/scaling/levante_campaign_2026-07-24.md:368:SECOND CANDIDATE ALSO REFUTED (`scripts/tmp/probe_sharded_overhead.py`,
docs/performance/scaling/levante_campaign_2026-07-24.md-369-job 26458553): the sharded formulation does NOT do measurably more work.
docs/performance/scaling/levante_campaign_2026-07-24.md-370-Timing the SHARDED step on a 1-device mesh (all the padding, band-edge and
docs/performance/scaling/levante_campaign_2026-07-24.md-371-v-row-reconstruction machinery present, ppermutes self-to-self so no real
docs/performance/scaling/levante_campaign_2026-07-24.md-372-traffic) against the UNSHARDED step at the identical tile:
docs/performance/scaling/levante_campaign_2026-07-24.md-373-
--
docs/performance/scaling/levante_campaign_2026-07-24.md-383-divided by the message count is **83 us/message at nd2 and 73 us at nd4**,
docs/performance/scaling/levante_campaign_2026-07-24.md-384-versus **17.8 us** for the same collective measured in isolation — an in-
docs/performance/scaling/levante_campaign_2026-07-24.md-385-context cost 4-5x the best case. That is consistent with EXPOSED,
docs/performance/scaling/levante_campaign_2026-07-24.md-386-un-overlapped communication rather than raw wire time.
docs/performance/scaling/levante_campaign_2026-07-24.md-387-
docs/performance/scaling/levante_campaign_2026-07-24.md:388:THIRD CANDIDATE REFUTED, AND IT IDENTIFIES THE MECHANISM (job 26458930).
docs/performance/scaling/levante_campaign_2026-07-24.md-389-If the residual were communication the scheduler is currently hiding work
docs/performance/scaling/levante_campaign_2026-07-24.md-390-behind, DISABLING XLA's latency-hiding scheduler would hurt. It does not:
docs/performance/scaling/levante_campaign_2026-07-24.md-391-
docs/performance/scaling/levante_campaign_2026-07-24.md-392-| arm | nd2 | nd4 | vs default |
docs/performance/scaling/levante_campaign_2026-07-24.md-393-|---|---|---|---|
--
docs/performance/scaling/levante_campaign_2026-07-24.md-454-genuinely PER-ITERATION, not a constant misattributed to iterations. But
docs/performance/scaling/levante_campaign_2026-07-24.md-455-the nd4 intercept EXCEEDS the measured 144-row compute term (14.63 ms) by
docs/performance/scaling/levante_campaign_2026-07-24.md-456-**1.48 ms**, so a fixed non-PCG overhead does exist and the 60 iterations
docs/performance/scaling/levante_campaign_2026-07-24.md-457-do NOT explain the entire residual.
docs/performance/scaling/levante_campaign_2026-07-24.md-458-
docs/performance/scaling/levante_campaign_2026-07-24.md:459:CODEX OBJECTION (a) TESTED AND REFUTED (job 26459817). Rather than divide
docs/performance/scaling/levante_campaign_2026-07-24.md-460-the nd1 slope by 4, measure the per-iteration slope directly at each tile
docs/performance/scaling/levante_campaign_2026-07-24.md-461-on ONE device:
docs/performance/scaling/levante_campaign_2026-07-24.md-462-
docs/performance/scaling/levante_campaign_2026-07-24.md-463-| tile | measured us/iter |
docs/performance/scaling/levante_campaign_2026-07-24.md-464-|---|---|
--
docs/performance/scaling/levante_campaign_2026-07-24.md-640-
docs/performance/scaling/levante_campaign_2026-07-24.md-641-Per-step time is flat 100 -> 600 steps (12.81 -> 12.74 ms). Both arms' heat
docs/performance/scaling/levante_campaign_2026-07-24.md-642-drift grows ~linearly and is similar between them, consistent with the
docs/performance/scaling/levante_campaign_2026-07-24.md-643-shared baroclinic/tracer path dominating it.
docs/performance/scaling/levante_campaign_2026-07-24.md-644-
docs/performance/scaling/levante_campaign_2026-07-24.md:645:MECHANISM CONFIRMED FROM A THIRD ANGLE. The wide-halo arm records its own
docs/performance/scaling/levante_campaign_2026-07-24.md-646-message census: **120 standard barotropic messages/step -> 4** (n_loop=30
docs/performance/scaling/levante_campaign_2026-07-24.md-647-substeps, stencil reach 3, one fixed wide exchange per chunk). A 30x cut in
docs/performance/scaling/levante_campaign_2026-07-24.md-648-barotropic exchanges is precisely why it wins where sync dominates, and it
docs/performance/scaling/levante_campaign_2026-07-24.md-649-is the SAME quantity the single_reduce analysis isolated as the half it
docs/performance/scaling/levante_campaign_2026-07-24.md-650-could not touch (44.1 us/iter of matvec halo). Three independent
--
docs/performance/scaling/levante_campaign_2026-07-24.md-982-
docs/performance/scaling/levante_campaign_2026-07-24.md-983-## Cube optimisation: bounded BEFORE implementing — and the bound killed the plan
docs/performance/scaling/levante_campaign_2026-07-24.md-984-
docs/performance/scaling/levante_campaign_2026-07-24.md-985-Directive was to push the cube toward its limit. Codex round-16 defined
docs/performance/scaling/levante_campaign_2026-07-24.md-986-the strategy first (per the standing pre-implementation rule) and its
docs/performance/scaling/levante_campaign_2026-07-24.md:987:cheapest-bound step then REFUTED the intervention I was about to build.
docs/performance/scaling/levante_campaign_2026-07-24.md-988-
docs/performance/scaling/levante_campaign_2026-07-24.md-989-**Codex corrections to my reading:**
docs/performance/scaling/levante_campaign_2026-07-24.md-990-* My "88 SendRecv / 4 rounds = 22 exchanges" was wrong. The kt=2 tile pad
docs/performance/scaling/levante_campaign_2026-07-24.md-991-  emits **16 phases per logical scalar pad** (4 edges + 4 guards + 4
docs/performance/scaling/levante_campaign_2026-07-24.md-992-  diagonals + 4 corner slivers) for serial-exact offset/corner handling
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1531-   CLOSED as not-worth-building, with receipts. (Also moot for the SPMD
docs/performance/scaling/levante_campaign_2026-07-24.md-1532-   lane: jax equal-shard sharding would need padding to the max band,
docs/performance/scaling/levante_campaign_2026-07-24.md-1533-   returning exactly the imbalance removed.)
docs/performance/scaling/levante_campaign_2026-07-24.md-1534-
docs/performance/scaling/levante_campaign_2026-07-24.md-1535-   *Expensive half — gather/scatter compaction: MEASURED, and the audit's
docs/performance/scaling/levante_campaign_2026-07-24.md:1536:   "~2x" is REFUTED* (`bench_gather_vs_slice_stencil.py`, job 26479884,
docs/performance/scaling/levante_campaign_2026-07-24.md-1537-   A100 f32, correctness self-checked). Per-cell gather penalty for a
docs/performance/scaling/levante_campaign_2026-07-24.md-1538-   5-point Laplacian vs the dense sliced version: **1.40-1.77x**, so
docs/performance/scaling/levante_campaign_2026-07-24.md-1539-   compaction wins only when wet_fraction < 0.56-0.72 (size-dependent).
docs/performance/scaling/levante_campaign_2026-07-24.md-1540-   At the REAL global-ocean wet fraction (~0.71), packed-gather is a net
docs/performance/scaling/levante_campaign_2026-07-24.md-1541-   LOSS on the full LL576 grid (ratio 1.16) and a wash at the nd4 tile
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1628-  METIS was NOT tested).
docs/performance/scaling/levante_campaign_2026-07-24.md-1629-* **`block:cyclic` stays mandatory on packed CPU lanes** (E/D = 1.61x at
docs/performance/scaling/levante_campaign_2026-07-24.md-1630-  a byte-identical partition). NOTE the second `--distribution` field is
docs/performance/scaling/levante_campaign_2026-07-24.md-1631-  the INTRA-NODE (socket) distribution — both arms place ranks on nodes
docs/performance/scaling/levante_campaign_2026-07-24.md-1632-  identically; the swing is socket-level. Mechanism (per-socket
docs/performance/scaling/levante_campaign_2026-07-24.md:1633:  memory-bandwidth balance) PLAUSIBLE, consistent with the np16 Milan
docs/performance/scaling/levante_campaign_2026-07-24.md-1634-  2.13x receipt; never instrumented with bandwidth counters.
docs/performance/scaling/levante_campaign_2026-07-24.md-1635-* Caveats: timing-only receipt — no parity/conservation gate ran in
docs/performance/scaling/levante_campaign_2026-07-24.md-1636-  these arms, and the CPU nodes emit `UCX WARN transports
docs/performance/scaling/levante_campaign_2026-07-24.md-1637-  'cuda_copy','cuda_ipc','gdr_copy' are not available` (the _env.sh GPU
docs/performance/scaling/levante_campaign_2026-07-24.md-1638-  UCX_TLS list on a CPU node; UCX falls back to rc/sm — cosmetic for
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1643-f32, sfc partition (padded-128 reorder), lloyd=0 LABELLED SYNTHETIC
docs/performance/scaling/levante_campaign_2026-07-24.md-1644-scaling mesh, executed padded n_cells = 2,621,568 (natural 2,621,442),
docs/performance/scaling/levante_campaign_2026-07-24.md-1645-L26; steps 12 / warmup 3; physics=none dynamics-only bench. Provenance:
docs/performance/scaling/levante_campaign_2026-07-24.md-1646-np64 and np128 rows record `git_sha: 7151d12a1`; the np32 row's field
docs/performance/scaling/levante_campaign_2026-07-24.md-1647-reads `unknown` — same allocation, same submitted script, so the same
docs/performance/scaling/levante_campaign_2026-07-24.md:1648:binary is PLAUSIBLE but that row stays non-reproduction-grade on its
docs/performance/scaling/levante_campaign_2026-07-24.md-1649-own (codex r20/r21):
docs/performance/scaling/levante_campaign_2026-07-24.md-1650-
docs/performance/scaling/levante_campaign_2026-07-24.md-1651-| GPUs | cells/GPU | ms/step | GC/s (cell-levels) |
docs/performance/scaling/levante_campaign_2026-07-24.md-1652-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-1653-| 32 | 81.9k | 12.47 | 5.47 |
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1678-   sets vs SAME-JOB solo controls bracketing phase B (solo before AND
docs/performance/scaling/levante_campaign_2026-07-24.md-1679-   after — BRACKETED, not fully counterbalanced; a penalty's attribution
docs/performance/scaling/levante_campaign_2026-07-24.md-1680-   to fabric vs placement/drift needs the per-step nodelist table +
docs/performance/scaling/levante_campaign_2026-07-24.md-1681-   follow-up). steps=5000 so the stepping window
docs/performance/scaling/levante_campaign_2026-07-24.md-1682-   (~60 s) dwarfs launch skew; per-arm `SLURM_STEP_NODELIST` +
docs/performance/scaling/levante_campaign_2026-07-24.md:1683:   wall-clock brackets logged as overlap evidence. CONFIRM bar:
docs/performance/scaling/levante_campaign_2026-07-24.md-1684-   max(replica) <= 1.10x mean(solo) => guaranteed aggregate >= 3.64x the
docs/performance/scaling/levante_campaign_2026-07-24.md-1685-   32-GPU solo rate (>= 19.9 GC/s if solo reproduces 5.47) = ~3.3x the
docs/performance/scaling/levante_campaign_2026-07-24.md:1686:   observed 128-GPU single-trajectory rate. REFUTE: replica slowdown
docs/performance/scaling/levante_campaign_2026-07-24.md-1687-   >10 % = a CO-EXECUTION penalty, quantified per replica — its
docs/performance/scaling/levante_campaign_2026-07-24.md-1688-   attribution (fabric contention vs placement/topology vs drift) is a
docs/performance/scaling/levante_campaign_2026-07-24.md-1689-   follow-up, not a conclusion of this job.
docs/performance/scaling/levante_campaign_2026-07-24.md-1690-2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
docs/performance/scaling/levante_campaign_2026-07-24.md-1691-   (81.9k/41.0k/20.5k cells/GPU) on the SAME lloyd=0 family, same sfc +
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1743-
docs/performance/scaling/levante_campaign_2026-07-24.md-1744-Distribution verified against the masquerade trap: result rows carry
docs/performance/scaling/levante_campaign_2026-07-24.md-1745-`n_ranks: 512` (the JSON's `metadata.process_count: 1` is the jax-LOCAL
docs/performance/scaling/levante_campaign_2026-07-24.md-1746-count on this mpi4jax lane, not the world size). 128->256 is
#!/bin/bash -l
#SBATCH --job-name=ll_bound
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=ll_bound.%j.log
# Single-device SAME-PER-DEVICE-SIZE baselines to complete the calibrated
# T_bound for the LL2048 rows (roofline recipe: the nd=1 time must be at
# the PER-DEVICE tile, not the global grid — passing the global one gives
# measured/bound < 1, the known tell).  Tiles: 16x4096 (the @128 band),
# 32x4096 (the @64 band).  APPROXIMATION, recorded: an nd=1 run includes
# the pole tiles, so its operator mix slightly OVERCOUNTS an interior
# band's compute -> the bound is conservative (biased high).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll_bound_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
run_arm () { # nlat x64 tag
  echo "=== nd=1 LL$1x4096 x64=$2 ($3) ==="
  JAX_ENABLE_X64=$2 srun --ntasks=1 --gpus=1 --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --n-devices 1 --mode strong --n-lat "$1" --n-lon 4096 --nlev 26 \
      --steps 12 --warmup 3 \
      --out "$OUTDIR/$3.jsonl" || { echo "$3 FAILED"; rc=1; }
}
run_arm 16 0 tile16_f32
run_arm 32 0 tile32_f32
run_arm 16 1 tile16_f64
echo "=== RESULTS ==="
for T in tile16_f32 tile32_f32 tile16_f64; do
  "$PY" -c "
import json,math,sys
try:
    d=json.loads(open('$OUTDIR/$T.jsonl').readline())
    ms=d['steady_median_ms']; assert math.isfinite(ms) and ms>0
except Exception as e:
    print('$T: MISSING/INVALID ->', e); sys.exit(1)
print(f'$T: {ms:8.3f} ms')" || { echo "$T receipt invalid"; rc=1; }
done
echo "DONE rc=$rc"; exit $rc
    ``hydrostatic_to_cgrid`` materialised the full ``(n_lat, n_lon, nlev)``
    state on every process before sharding).

    Bit-identical to
    ``shard_state_atm_latlon(hydrostatic_to_cgrid(held_suarez_init_latlon(
    grid, sigma), grid), mesh)`` for the flat-terrain case — gated by
    ``tests/parallel/test_atm_latlon_bandlocal_build.py``.  The staggered
    ``v`` is created directly as its sharded ``v_lower`` layout (the dropped
    north pole-wall face is identically zero in this at-rest IC, exactly what
    ``gather_state_atm_latlon`` re-appends).

    One deliberate exception: the 2-D lowest-level temperature perturbation
    (``jax.random.normal`` over ``(n_lat, n_lon)``) is evaluated in full on
    every process — identical threefry streams cannot be row-sliced without
    evaluating the whole field, and at bench scale it is O(10 MB) vs the
    O(GB) 3-D leaves the callback path avoids.

    Flat terrain only (the bench IC): there is deliberately no ``phis``
    parameter — the topography variant of ``held_suarez_init_latlon`` would
    need its own band-local surface-pressure callback; extend explicitly
    rather than reuse this builder.
    """
    from legoesm import constants
    from legoesm.core.precision import get_policy

    _dtype = get_policy().storage
    if p_s_init is None:
        p_s_init = constants.p_ref

    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = sigma_coord.n_levels

    # 2-D seed field, exact expression of held_suarez_init_latlon
    key = jax.random.PRNGKey(seed)
    pert2d = jax.random.normal(key, (n_lat, n_lon), dtype=_dtype) * jnp.asarray(
        perturbation_amplitude, dtype=_dtype
    )

    def _slice_shape(gshape, idx):
        return tuple(len(range(*sl.indices(n))) for sl, n in zip(idx, gshape))

    def _make(gshape, cb):
        sharding = NamedSharding(mesh, P("lat", *((None,) * (len(gshape) - 1))))
        return jax.make_array_from_callback(gshape, sharding, cb)

    def _zeros_cb(gshape):
        return lambda idx: jnp.zeros(_slice_shape(gshape, idx), dtype=_dtype)

    def _T_cb(idx):
        shape = _slice_shape((n_lat, n_lon, nlev), idx)
        # EXACT expression of held_suarez_init_latlon (``ones * T_init`` then
        # ``.at[:, :, -1].add(pert)``) so promotion semantics match for
        # strongly-typed ``T_init`` too, not just Python floats.
        block = jnp.ones(shape, dtype=_dtype) * T_init
        return block.at[:, :, -1].add(pert2d[idx[0], idx[1]])

    def _ps_cb(idx):
        shape = _slice_shape((n_lat, n_lon), idx)
        # held_suarez_init_latlon with phis=None: p_s_init * exp(-0/(R_d T))
        phis_block = jnp.zeros(shape, dtype=_dtype)
        return (p_s_init * jnp.exp(
            -phis_block / (constants.R_d * T_init))).astype(_dtype)

    sh_u = (n_lat, n_lon + 1, nlev)
    sh_vlow = (n_lat, n_lon, nlev)   # sharded layout: pole-wall face dropped
    sh_T = (n_lat, n_lon, nlev)
    sh_2d = (n_lat, n_lon)
    return CGridLatLonHydrostaticState(
        u=_make(sh_u, _zeros_cb(sh_u)),
        v=_make(sh_vlow, _zeros_cb(sh_vlow)),
        T=_make(sh_T, _T_cb),
        p_s=_make(sh_2d, _ps_cb),
        phis=_make(sh_2d, _zeros_cb(sh_2d)),
        tracers={},
    )


def gather_state_atm_latlon(
    state: CGridLatLonHydrostaticState, mesh,
) -> CGridLatLonHydrostaticState:
    """Inverse of :func:`shard_state_atm_latlon`: replicate every leaf and
    rebuild the full ``(n_lat+1, ...)`` ``v`` by re-appending the zero north
    pole-wall face. Bit-comparable to the single-device state (whose top v-face
    is the pole wall == 0).

    Multi-controller (route-B ``jax.distributed``, mesh spanning processes):
    replication routes through a jit-compiled identity instead of
    ``device_put`` (see :func:`legoesm.parallel.latlon_spmd.replicate_leaf`,
    the primitive shared with the ocean gather); the single-process path is
    byte-unchanged."""
    from legoesm.parallel.latlon_spmd import replicate_leaf
            polar_mask=pmask, polar_mask_v=pmaskv,
            pole_v_bc_masks=spmd_pole_end_masks(),
        )
        return out._replace(v=to_vface_lower(out.v)), ps_out

    return band_step


def _dtype_sig(tree) -> tuple:
    """Trace-time carry signature of a pytree: the pytree STRUCTURE plus
    every array leaf's ``(shape, dtype, weak_type)``.

    ``lax.scan`` requires carry-in == carry-out in ALL of structure, shape,
    dtype and weak type — dtype strings alone would mark a step that flips
    weak typing (or shape) while preserving dtypes as "stable" and then
    fail inside the scan lowering instead of being absorbed by the unroll
    (codex M3b review).  Non-array leaves contribute through the treedef.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return (str(treedef),
            tuple((tuple(leaf.shape), str(leaf.dtype),
                   bool(getattr(leaf, "weak_type", False)))
                  for leaf in leaves if hasattr(leaf, "dtype")))


def unroll_to_dtype_fixed_point(step1, state, n_left: int):
    """Unroll ``step1`` applications until the state's dtype signature is a
    FIXED POINT of the step (bounded by ``n_left``); returns
    ``(state, n_left_remaining)``.

    PUBLIC (M3b): shared by this module's lat-band segments and the tiled
    cube segment (``tiled_step_adapter.scan_tiled_cc_steps``) — any
    ``lax.scan``-of-a-step needs carry-in == carry-out dtypes, and this is
    the ONE place that trace-time unroll lives (no re-derivation).

    Why: ``lax.scan`` needs carry-in == carry-out dtypes.  A mixed-precision
    IC (f32 grid/init-derived leaves beside the step's f64 sources under
    x64 — the f64 sigma-coordinate arrays and the strong-f64
    ``jnp.asarray(dt)`` operand) promotes over the first stepS exactly as
    the per-step Python loop absorbs silently: observed on the bench IC
    (jobs 8916406/8916740), ``p_s`` promotes f32->f64 in step 1 and
    ``u/v/T`` follow in step 2 by mixing the now-f64 ``p_s`` — the fixed
    point can take MORE than one application, and no role-based cast can
    express it (the default f32 policy no-ops while the promotion is
    mixed-LEAF arithmetic).

    ``jax.eval_shape`` probes the step's output dtypes ABSTRACTLY at trace
    time (zero FLOPs), so exactly the needed number of steps is unrolled —
    NONE for an already dtype-stable state (the parity-gate f64 states scan
    all ``n_steps``).  The unroll count is a trace-time constant baked into
    the compiled program: every segment call executes the same
    ``k unrolled + scan(n_left)`` schedule with ``k + n_left == n_steps``.
    Zero extra casts, zero numerical difference vs the per-step lane;
    strictly monotone leaf promotion over a finite dtype lattice guarantees
    termination, and the ``n_left`` bound caps the unroll at the segment
    length (a 1-step segment simply runs its single step unrolled).
    """
    while (n_left > 0
           and _dtype_sig(jax.eval_shape(step1, state)) != _dtype_sig(state)):
        state = step1(state)
        n_left -= 1
    return state, n_left


def _refuse_unsupported_spmd_config(model) -> None:
    """Dispatch-hardening shared by the step + segment factories: only a
    non-fold lat-lon grid with an SPMD-safe mass path is supported.  Fail
    LOUD rather than silently mis-fold / band-local-sum."""
    fold = getattr(model.grid, "fold", None)
    if fold is not None and bool(getattr(fold, "is_active", False)):
        raise NotImplementedError(
            "atm lat-band SPMD: tripole north-fold is a follow-up.")
    if getattr(model.config, "anchor_mass_to_initial", False):
        raise NotImplementedError(
            "atm lat-band SPMD: anchor_mass_to_initial uses a band-local "
            "jnp.sum(p_s*area) target that is not yet SPMD-routed; disable it "
            "or use fix_mass with the pre-state (psum'd) path.")


@contextmanager
def _latlon_spmd_armed(mesh):
        steady_median_ms=round(med, 4),
        cells=n_lat * args.n_lon * args.nlev,
    )
    if seg_n > 0:
        # Segment lane: unit 0 = the first BLOCK (includes the scan
        # compile); per-step numbers derive from whole blocks.
        rec.update(
            compile_ms=round(per_block_ms[0], 1),
            steady_min_ms=round(float(np.min(steady)), 2),
            per_step_ms=[round(x, 2) for x in per_step_ms],
            per_block_ms=[round(x, 2) for x in per_block_ms],
        )
    else:
        # Default lane: the timed_scan_blocks metrics (fused_step_ms,
        # step_latency_ms, block_ms, parallel_block_ms, rank_imbalance, ...
        # — the M1 measurement contract).
        rec.update(**timing)
    # Flat aggregator-compatible identity + metric fields: without a
    # top-level ``sypd``/``grid_type`` this lane's rows are invisible to
    # aggregate_bcw_scaling.py → empty SYPD panels in the CPU-vs-GPU plots.
    rec.update(
        grid_type="latlon",
        resolution=n_lat,
        n_levels=args.nlev,
        precision="float64" if jax.config.jax_enable_x64 else "float32",
        physics_level=args.physics,
        backend=jax.default_backend(),
        **tidy_throughput_fields(
            dt_seconds=args.dt, time_per_step_ms=med,
            total_cells=n_lat * args.n_lon * args.nlev),
    )
    # Increment-2 accounting fields (audit items 4/8), flat for aggregators.
    rec.update(**comm_rec, **bound_rec)
    if not valid:
        # Throughput derived from a diverging trajectory is not a
        # measurement: null it explicitly so aggregators cannot ingest it
        # even if they miss the valid flag (codex batch4) — including the
        # M1 fused/per-step-time keys plotters fall back to
        # (time_per_step_ms, steady_min_ms) and the med-derived bound
        # diagnostics (already null via the _measured_med feed above).
        for _k in ("sypd", "mcells_per_s", "time_per_step_ms",
                   "steady_min_ms", "fused_step_ms", "step_latency_ms",
                   "t_bound_ms", "measured_over_bound"):
            if _k in rec:
                rec[_k] = None
    from legoesm.parallel.early_init import nccl_transport_report
    _nccl_report = nccl_transport_report()
    rec["metadata"] = annotate_incomplete(scaling_metadata(
        grid="latlon",
        component="atmosphere",
        resolution=f"{n_lat}x{args.n_lon}",
        n_levels=args.nlev,
        precision="float64" if jax.config.jax_enable_x64 else "float32",
        n_gpus=(nd if jax.default_backend() in ("gpu", "cuda", "rocm")
                else 0),
        decomposition="band" if nd > 1 else "none",
        # cells_per_rank is per PROCESS (n_ranks semantics); the per-device
        # share lives in extra.cells_per_device — a single-process 4-device
        # SPMD run has 1 rank owning ALL cells (codex finding 3).
        cells_per_rank=(n_lat * args.n_lon * args.nlev)
        // max(jax.process_count(), 1),
        scaling_kind=args.mode,
        extra={
            "physics": args.physics,
            "steps": args.steps,
            "warmup": args.warmup,

codex
The arithmetic implementation is straightforward; the remaining audit focus is whether the byte probe is measuring the executed payload rather than an HLO formatting artifact, and whether the f64 extrapolation is valid.
exec
/usr/bin/bash -lc 'JAX_PLATFORMS=cpu /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python scripts/tmp/_probe_latlon_halo_census.py 8 512 1024 26' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 11530ms:
{"cp_ops_with_shape": 41, "cp_result_bytes_per_dev_per_step": 4635408, "nd": 8, "n_lat": 512, "n_lon": 1024, "nlev": 26, "rows_per_dev": 64, "census_backend": "virtual-cpu (forced host platform)", "census": {"collective_permute": 41, "all_reduce": 1, "all_gather": 0, "all_to_all": 0, "reduce_scatter": 0, "total": 42}}

exec
/usr/bin/bash -lc "JAX_PLATFORMS=cpu /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python scripts/tmp/_probe_latlon_halo_census.py 16 512 1024 26
JAX_PLATFORMS=cpu /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python scripts/tmp/_probe_latlon_halo_census.py 8 512 2048 26
sed -n '400,485p' tests/bench/test_scaling_metadata.py
rg -n \"get_policy|class.*Precision|storage\" packages/core/legoesm/core packages/core/legoesm -g '*precision*.py' | sed -n '1,120p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 12342ms:
{"cp_ops_with_shape": 41, "cp_result_bytes_per_dev_per_step": 4635408, "nd": 16, "n_lat": 512, "n_lon": 1024, "nlev": 26, "rows_per_dev": 32, "census_backend": "virtual-cpu (forced host platform)", "census": {"collective_permute": 41, "all_reduce": 1, "all_gather": 0, "all_to_all": 0, "reduce_scatter": 0, "total": 42}}
{"cp_ops_with_shape": 41, "cp_result_bytes_per_dev_per_step": 9263888, "nd": 8, "n_lat": 512, "n_lon": 2048, "nlev": 26, "rows_per_dev": 64, "census_backend": "virtual-cpu (forced host platform)", "census": {"collective_permute": 41, "all_reduce": 1, "all_gather": 0, "all_to_all": 0, "reduce_scatter": 0, "total": 42}}
    count_collective_permutes' synthetic HLO test above."""
    import jax.numpy as jnp
    with _cpu_compile():
        assert md.hlo_collective_permutes(lambda x: x + 1, jnp.arange(4.0)) == 0

        def _boom(x):
            raise RuntimeError("unlowerable")
        assert md.hlo_collective_permutes(_boom, jnp.arange(4.0)) is None


def test_count_collectives_full_census_all_families():
    """Full census counts every collective family with the same op-call-form
    discipline: async ``-start`` once (``-done`` excluded), StableHLO
    underscore + optimized hyphen, and a config-header flag echo that merely
    CONTAINS an op name never inflates the count."""
    hlo = "\n".join([
        # config-header echo of XLA_FLAGS -> must NOT match (no op-call paren)
        "  // xla_gpu_collective_permute_combine_threshold_bytes=33554432",
        "  %a = collective-permute(%x)",             # permute (optimized)
        "  %b = collective_permute(%y)",             # permute (StableHLO)
        "  %c = collective-permute-done(%a)",        # async companion -> drop
        "  %r1 = all-reduce(%p)",                    # reduction (hyphen)
        "  %r2 = all_reduce_start(%q)",              # async reduction -> count once
        "  %r3 = all-reduce-done(%r2)",              # async companion -> drop
        "  %g = all-gather(%z)",                     # all-gather
        "  %a2a = all-to-all(%w)",                   # all-to-all
        "  %rs = reduce-scatter(%v)",                # reduce-scatter
    ])
    c = md.count_collectives(hlo)
    assert c["collective_permute"] == 2
    assert c["all_reduce"] == 2          # hyphen op + async-start (done dropped)
    assert c["all_gather"] == 1
    assert c["all_to_all"] == 1
    assert c["reduce_scatter"] == 1
    assert c["total"] == 7
    # permute family stays bit-identical to the canonical scalar helper
    assert c["collective_permute"] == md.count_collective_permutes(hlo)

    empty = md.count_collectives("no collectives here")
    assert empty["total"] == 0
    assert set(empty) == {"collective_permute", "all_reduce", "all_gather",
                          "all_to_all", "reduce_scatter", "total"}


def test_census_does_not_false_drop_ops_with_done_in_metadata():
    """Regression (codex): the ``-done`` async COMPANION is excluded by the
    regex structurally (``op-done(`` never matches ``op(?:[_-]start)?\\(``), so
    a line that merely CONTAINS the substring "done" elsewhere — an XLA
    metadata op_name, a ``%done_*`` SSA name — must STILL be counted.  A blunt
    ``"done" not in line`` filter would false-drop these to zero."""
    hlo = "\n".join([
        '  %r = all-reduce(%p), metadata={op_name="jit(step)/done_stage/psum"}',
        '  %done_mass = f32[] collective-permute(%q)',
        '  %g = all-gather(%z), metadata={op_name="reduce_done/x"}',
    ])
    c = md.count_collectives(hlo)
    assert c["all_reduce"] == 1        # NOT dropped despite "done" in metadata
    assert c["collective_permute"] == 1  # NOT dropped despite %done_ SSA name
    assert c["all_gather"] == 1
    # canonical permute helper is fixed by the same shared counter
    assert md.count_collective_permutes(hlo) == 1


def test_hlo_collective_census_lowers_and_is_error_safe():
    """Best-effort full-census probe: a collective-free fn -> all-zero dict;
    an unlowerable fn -> None (never raises)."""
    import jax.numpy as jnp
    with _cpu_compile():
        census = md.hlo_collective_census(lambda x: x + 1, jnp.arange(4.0))
        assert census is not None and census["total"] == 0

        def _boom(x):
            raise RuntimeError("unlowerable")
        assert md.hlo_collective_census(_boom, jnp.arange(4.0)) is None
packages/core/legoesm/runtime/precision.py:24:    get_policy,
packages/core/legoesm/runtime/precision.py:51:    "mixed_fp64_storage": PrecisionPolicy.mixed_fp64_storage,
packages/core/legoesm/runtime/precision.py:55:#: backend — e.g. not Apple Metal).  ``"mixed"`` keeps fp32 storage, so it is OK
packages/core/legoesm/runtime/precision.py:57:_FP64_STORAGE_MODES = frozenset({"fp64", "float64", "mixed_fp64_storage"})
packages/core/legoesm/runtime/precision.py:110:    if mode.strip().lower() in ("mixed", "mixed_fp64_storage"):
packages/core/legoesm/core/precision.py:3:Provides a unified precision policy system that controls storage, compute,
packages/core/legoesm/core/precision.py:17:>>> from legoesm.core.precision import PrecisionPolicy, get_policy, cast, const
packages/core/legoesm/core/precision.py:80:class PrecisionPolicy(NamedTuple):
packages/core/legoesm/core/precision.py:85:    storage : jnp.dtype
packages/core/legoesm/core/precision.py:96:    storage: jnp.dtype = jnp.float32
packages/core/legoesm/core/precision.py:105:            storage=jnp.float32,
packages/core/legoesm/core/precision.py:115:            storage=jnp.float64,
packages/core/legoesm/core/precision.py:129:            storage=jnp.float32,
packages/core/legoesm/core/precision.py:136:    def mixed_fp64_storage() -> PrecisionPolicy:
packages/core/legoesm/core/precision.py:137:        """Mixed mode with float64 storage, float32 compute.
packages/core/legoesm/core/precision.py:145:            storage=jnp.float64,
packages/core/legoesm/core/precision.py:175:    if jnp.float64 in (policy.storage, policy.compute,
packages/core/legoesm/core/precision.py:181:def get_policy() -> PrecisionPolicy:
packages/core/legoesm/core/precision.py:200:        policy = get_policy()
packages/core/legoesm/core/precision.py:202:        policy.storage, policy.compute, policy.accumulate, policy.control,
packages/core/legoesm/core/precision.py:225:        Role name -> precision mode string. Valid roles: storage, compute,
packages/core/legoesm/core/precision.py:234:    valid_roles = {"storage", "compute", "accumulate", "control"}
packages/core/legoesm/core/precision.py:291:    policy = get_policy()
packages/core/legoesm/core/precision.py:319:        Precision role: "storage", "compute", "accumulate", or "control".
packages/core/legoesm/core/precision.py:479:    """Decorator that casts inputs to compute dtype and output to storage dtype.
packages/core/legoesm/core/precision.py:488:        # Output is automatically cast to storage dtype
packages/core/legoesm/core/precision.py:494:            storage_dtype = resolve_dtype(module, "storage")
packages/core/legoesm/core/precision.py:508:            # Cast output back to storage dtype.
packages/core/legoesm/core/precision.py:509:            def _to_storage(leaf):
packages/core/legoesm/core/precision.py:513:                    return leaf.astype(storage_dtype)
packages/core/legoesm/core/precision.py:516:            return jax.tree.map(_to_storage, result)
packages/core/legoesm/core/precision.py:555:        "storage": jnp.float64, "compute": jnp.float64,
packages/core/legoesm/core/precision.py:3:Provides a unified precision policy system that controls storage, compute,
packages/core/legoesm/core/precision.py:17:>>> from legoesm.core.precision import PrecisionPolicy, get_policy, cast, const
packages/core/legoesm/core/precision.py:80:class PrecisionPolicy(NamedTuple):
packages/core/legoesm/core/precision.py:85:    storage : jnp.dtype
packages/core/legoesm/core/precision.py:96:    storage: jnp.dtype = jnp.float32
packages/core/legoesm/core/precision.py:105:            storage=jnp.float32,
packages/core/legoesm/core/precision.py:115:            storage=jnp.float64,
packages/core/legoesm/core/precision.py:129:            storage=jnp.float32,
packages/core/legoesm/core/precision.py:136:    def mixed_fp64_storage() -> PrecisionPolicy:
packages/core/legoesm/core/precision.py:137:        """Mixed mode with float64 storage, float32 compute.
packages/core/legoesm/core/precision.py:145:            storage=jnp.float64,
packages/core/legoesm/core/precision.py:175:    if jnp.float64 in (policy.storage, policy.compute,
packages/core/legoesm/core/precision.py:181:def get_policy() -> PrecisionPolicy:
packages/core/legoesm/core/precision.py:200:        policy = get_policy()
packages/core/legoesm/core/precision.py:202:        policy.storage, policy.compute, policy.accumulate, policy.control,
packages/core/legoesm/core/precision.py:225:        Role name -> precision mode string. Valid roles: storage, compute,
packages/core/legoesm/core/precision.py:234:    valid_roles = {"storage", "compute", "accumulate", "control"}
packages/core/legoesm/core/precision.py:291:    policy = get_policy()
packages/core/legoesm/core/precision.py:319:        Precision role: "storage", "compute", "accumulate", or "control".
packages/core/legoesm/core/precision.py:479:    """Decorator that casts inputs to compute dtype and output to storage dtype.
packages/core/legoesm/core/precision.py:488:        # Output is automatically cast to storage dtype
packages/core/legoesm/core/precision.py:494:            storage_dtype = resolve_dtype(module, "storage")
packages/core/legoesm/core/precision.py:508:            # Cast output back to storage dtype.
packages/core/legoesm/core/precision.py:509:            def _to_storage(leaf):
packages/core/legoesm/core/precision.py:513:                    return leaf.astype(storage_dtype)
packages/core/legoesm/core/precision.py:516:            return jax.tree.map(_to_storage, result)
packages/core/legoesm/core/precision.py:555:        "storage": jnp.float64, "compute": jnp.float64,

exec
/usr/bin/bash -lc "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python scripts/tmp/_probe_latlon_halo_census.py 8 512 1024 26
sed -n '80,210p' packages/core/legoesm/core/precision.py
rg -n -C 5 \"def ppermute|ppermute\\(\" packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py packages/core/legoesm/parallel/latlon_spmd.py packages -g '*.py' | rg -C 3 \"ppermute|collective\" | sed -n '1,360p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 6446ms:
{"cp_ops_with_shape": 41, "cp_result_bytes_per_dev_per_step": 4635408, "nd": 8, "n_lat": 512, "n_lon": 1024, "nlev": 26, "rows_per_dev": 64, "census_backend": "virtual-cpu (forced host platform)", "census": {"collective_permute": 41, "all_reduce": 1, "all_gather": 0, "all_to_all": 0, "reduce_scatter": 0, "total": 42}}
class PrecisionPolicy(NamedTuple):
    """Global precision configuration.

    Attributes
    ----------
    storage : jnp.dtype
        Dtype for state arrays in memory. Determines GPU memory footprint.
    compute : jnp.dtype
        Dtype for arithmetic in kernels (matmuls, stencils, tendencies).
    accumulate : jnp.dtype
        Dtype for reductions (global sums, budget integrals, norms).
        Must be >= compute width to avoid catastrophic cancellation.
    control : jnp.dtype
        Dtype for solvers, implicit systems, and diagnostics that
        require tight tolerances (semi-implicit, barotropic, EOS).
    """
    storage: jnp.dtype = jnp.float32
    compute: jnp.dtype = jnp.float32
    accumulate: jnp.dtype = jnp.float32
    control: jnp.dtype = jnp.float32

    @staticmethod
    def fp32() -> PrecisionPolicy:
        """All-float32 mode. Fastest, GPU-optimal."""
        return PrecisionPolicy(
            storage=jnp.float32,
            compute=jnp.float32,
            accumulate=jnp.float32,
            control=jnp.float32,
        )

    @staticmethod
    def fp64() -> PrecisionPolicy:
        """All-float64 mode. Full scientific reference."""
        return PrecisionPolicy(
            storage=jnp.float64,
            compute=jnp.float64,
            accumulate=jnp.float64,
            control=jnp.float64,
        )

    @staticmethod
    def mixed() -> PrecisionPolicy:
        """Mixed-precision mode.

        Storage and compute in float32 for GPU bandwidth.
        Accumulation and control in float64 for stability.
        """
        return PrecisionPolicy(
            storage=jnp.float32,
            compute=jnp.float32,
            accumulate=jnp.float64,
            control=jnp.float64,
        )

    @staticmethod
    def mixed_fp64_storage() -> PrecisionPolicy:
        """Mixed mode with float64 storage, float32 compute.

        State arrays are stored in float64 for maximum precision in
        I/O, checkpointing, and long-run accumulation.  Compute kernels
        run in float32 for GPU throughput.  Accumulation and control
        remain float64.
        """
        return PrecisionPolicy(
            storage=jnp.float64,
            compute=jnp.float32,
            accumulate=jnp.float64,
            control=jnp.float64,
        )


# Module-level recommended dtypes: maps (module_name, role) -> role override.
# When a module has a specific precision requirement, it overrides the global
# policy for that role. None means "use the global policy default".
_MODULE_OVERRIDES: dict[str, dict[str, str | None]] = {}

# The active global policy.
_ACTIVE_POLICY: list[PrecisionPolicy] = [PrecisionPolicy.fp32()]


# ---------------------------------------------------------------------------
# Policy management
# ---------------------------------------------------------------------------

def set_policy(policy: PrecisionPolicy) -> None:
    """Set the active global precision policy.

    Also enables JAX x64 if any dtype is float64.  On backends that
    lack float64 hardware (e.g. Apple Metal), x64 is still enabled
    because spectral solvers route computation to CPU and need float64
    there.  ``resolve_dtype`` handles clamping float64 → float32 for
    non-spectral code that runs on the default (Metal) device.
    """
    _ACTIVE_POLICY[0] = policy
    if jnp.float64 in (policy.storage, policy.compute,
                        policy.accumulate, policy.control):
        if not jax.config.jax_enable_x64:
            jax.config.update("jax_enable_x64", True)


def get_policy() -> PrecisionPolicy:
    """Return the active global precision policy."""
    return _ACTIVE_POLICY[0]


def validate_policy(policy: PrecisionPolicy | None = None) -> None:
    """Verify the active precision policy is actually achievable.

    On backends that lack float64 (e.g. Metal), float64 requests in the
    policy are silently clamped to float32 by ``resolve_dtype``, so the
    policy is always achievable — this function is a no-op in that case.

    Raises
    ------
    RuntimeError
        If the policy requires float64, the backend supports it, but
        JAX x64 mode is not enabled.
    """
    if policy is None:
        policy = get_policy()
    needs_x64 = jnp.float64 in (
        policy.storage, policy.compute, policy.accumulate, policy.control,
    )
    if not needs_x64:
        return
    if not _backend().supports_float64():
        # Backend cannot do float64; resolve_dtype will clamp to float32.
        return
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
packages/core/legoesm/parallel/latlon_spmd.py-107-    perm_to_west, perm_to_east = latlon_lon_ring_perms(p_lon)
packages/core/legoesm/parallel/latlon_spmd.py-108-    # My EAST ghost = east neighbour's west edge (sources send WEST edges to
packages/core/legoesm/parallel/latlon_spmd.py-109-    # their west neighbour); my WEST ghost = west neighbour's east edge.
packages/core/legoesm/parallel/latlon_spmd.py:110:    east_ghost = jax.lax.ppermute(f[:, :halo], "lon", perm_to_west)
packages/core/legoesm/parallel/latlon_spmd.py:111:    west_ghost = jax.lax.ppermute(f[:, -halo:], "lon", perm_to_east)
packages/core/legoesm/parallel/latlon_spmd.py-112-    return jnp.concatenate([west_ghost, f, east_ghost], axis=1)
packages/core/legoesm/parallel/latlon_spmd.py-113-
packages/core/legoesm/parallel/latlon_spmd.py-114-
--
packages/core/legoesm/parallel/latlon_spmd.py-121-    sharded directly; it is carried as ``v_lower = v[:n_lat]`` (``n_lat`` rows,
packages/core/legoesm/parallel/latlon_spmd.py-122-    divisible by ``N``). Each band's NORTH boundary face is the next band's
packages/core/legoesm/parallel/latlon_spmd.py-123-    ``v_lower[0]`` (= the shared global interface row), lifted down via
packages/core/legoesm/parallel/latlon_spmd.py:124:    ``ppermute(..., perm_north)``; the north-most band has no neighbour there and
packages/core/legoesm/parallel/latlon_spmd.py-125-    receives the pole-wall zero (the ppermute non-target). Pure array core (no
packages/core/legoesm/parallel/latlon_spmd.py-126-    Field/state coupling) shared by the ocean and atmosphere lat-band SPMD steps
packages/core/legoesm/parallel/latlon_spmd.py-127-    so the v-stagger numerics are written ONCE (factored from the ocean step's
packages/core/legoesm/parallel/latlon_spmd.py-128-    ``_reconstruct_v`` closure). AD-safe: ``ppermute`` is self-transposing.
packages/core/legoesm/parallel/latlon_spmd.py-129-
--
packages/core/legoesm/parallel/latlon_spmd.py-135-
--
packages/core/legoesm/parallel/latlon_spmd.py-137-    -------
packages/core/legoesm/parallel/latlon_spmd.py-138-    array ``(n_lat_band + 1, n_lon[, nlev])`` — the band's full v-faces.
packages/core/legoesm/parallel/latlon_spmd.py-139-    """
packages/core/legoesm/parallel/latlon_spmd.py:140:    boundary = jax.lax.ppermute(v_lower[0:1], axis, perm_north)
packages/core/legoesm/parallel/latlon_spmd.py-141-    return jnp.concatenate([v_lower, boundary], axis=0)
packages/core/legoesm/parallel/latlon_spmd.py-142-
packages/core/legoesm/parallel/latlon_spmd.py-143-
--
packages/core/legoesm/parallel/latlon_spmd.py-198-            widths.append(w)
packages/core/legoesm/parallel/latlon_spmd.py-199-            flats.append(row.reshape(1, w))
packages/core/legoesm/parallel/latlon_spmd.py-200-        buf = jnp.concatenate(flats, axis=1)
packages/core/legoesm/parallel/latlon_spmd.py-201-        # ONE ppermute for the whole dtype group (north band receives 0).
packages/core/legoesm/parallel/latlon_spmd.py:202:        recv = jax.lax.ppermute(buf, axis, perm_north)
packages/core/legoesm/parallel/latlon_spmd.py-203-        off = 0
packages/core/legoesm/parallel/latlon_spmd.py-204-        for k, i in enumerate(idxs):
packages/core/legoesm/parallel/latlon_spmd.py-205-            w = widths[k]
--
packages/core/legoesm/parallel/latlon_spmd.py-249-        boundary = u_left[:, 0:1]
packages/core/legoesm/parallel/latlon_spmd.py-250-    else:
packages/core/legoesm/parallel/latlon_spmd.py-251-        perm_to_west, _ = latlon_lon_ring_perms(p_lon)
packages/core/legoesm/parallel/latlon_spmd.py:252:        boundary = jax.lax.ppermute(u_left[:, 0:1], axis, perm_to_west)
packages/core/legoesm/parallel/latlon_spmd.py-253-    return jnp.concatenate([u_left, boundary], axis=1)
packages/core/legoesm/parallel/latlon_spmd.py-254-
packages/core/legoesm/parallel/latlon_spmd.py-255-
packages/core/legoesm/parallel/latlon_spmd.py-256-def to_uface_left(u_full):
packages/core/legoesm/parallel/latlon_spmd.py-257-    """Inverse of :func:`reconstruct_uface_left`: drop the east seam column
--
packages/core/legoesm/parallel/latlon_spmd.py-387-    # 1. E/W-extend the edge rows by 2h (one lon ring ppermute pair).
packages/core/legoesm/parallel/latlon_spmd.py-388-    ext = lon_ring_ghosts_spmd(edge, mesh, halo=2 * h)  # (halo, w+4h[, lev])
packages/core/legoesm/parallel/latlon_spmd.py-389-    # 2. ONE antipodal ppermute over the lon ring (shift by p_lon/2 is a
packages/core/legoesm/parallel/latlon_spmd.py-390-    # bijection, and c' != c for every even p_lon >= 2).
packages/core/legoesm/parallel/latlon_spmd.py-391-    perm_anti = [(s, (s + p_lon // 2) % p_lon) for s in range(p_lon)]
packages/core/legoesm/parallel/latlon_spmd.py:392:    recv = jax.lax.ppermute(ext, "lon", perm_anti)
packages/core/legoesm/parallel/latlon_spmd.py-393-    # 3. lat-mirror + sign (the _pole_fold row flip), then the window
packages/core/legoesm/parallel/latlon_spmd.py-394-    # column map (derivation above; seam-straddling windows mix branches).
packages/core/legoesm/parallel/latlon_spmd.py-395-    sign = -1.0 if negate else 1.0
--
packages/core/legoesm/parallel/latlon_spmd.py-397-    t = jnp.arange(w + 2 * h)
--
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
--
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
--
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
--
packages/core/legoesm/parallel/latlon_spmd.py-705-                              fields[i][-halo:].reshape(halo, w)))
packages/core/legoesm/parallel/latlon_spmd.py-706-            south_buf = jnp.concatenate([s for s, _ in flats], axis=1)
packages/core/legoesm/parallel/latlon_spmd.py-707-            north_buf = jnp.concatenate([n for _, n in flats], axis=1)
packages/core/legoesm/parallel/latlon_spmd.py-708-            # ONE ppermute pair for the whole dtype group.
packages/core/legoesm/parallel/latlon_spmd.py:709:            north_recv = jax.lax.ppermute(south_buf, axis, perm_north)
packages/core/legoesm/parallel/latlon_spmd.py:710:            south_recv = jax.lax.ppermute(north_buf, axis, perm_south)
packages/core/legoesm/parallel/latlon_spmd.py-711-            off = 0
packages/core/legoesm/parallel/latlon_spmd.py-712-            for k, i in enumerate(idxs):
packages/core/legoesm/parallel/latlon_spmd.py-713-                w = widths[k]
--
--
packages/core/legoesm/parallel/cubesphere_exchange.py-508-
packages/core/legoesm/parallel/cubesphere_exchange.py-509-# ===================================================================
packages/core/legoesm/parallel/cubesphere_exchange.py-510-# Backend B: ppermute  (bandwidth-optimal for high resolution)
packages/core/legoesm/parallel/cubesphere_exchange.py-511-# ===================================================================
packages/core/legoesm/parallel/cubesphere_exchange.py-512-
packages/core/legoesm/parallel/cubesphere_exchange.py:513:def _make_exchange_ppermute(mesh, ndim, with_offsets=False):
packages/core/legoesm/parallel/cubesphere_exchange.py-514-    """Build a shard_map exchange using 4 rounds of ppermute.
packages/core/legoesm/parallel/cubesphere_exchange.py-515-
packages/core/legoesm/parallel/cubesphere_exchange.py-516-    When ``with_offsets`` is True the kernel applies the per-edge
packages/core/legoesm/parallel/cubesphere_exchange.py-517-    3-point Lagrange correction to each gathered strip — same numerics
--
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
--
packages/core/legoesm/parallel/cubesphere_exchange.py-1342-        # (table build asserts), so zeros init is dead weight only
--
packages/core/legoesm/parallel/cubesphere_exchange.py-1344-        halo_strips = jnp.zeros_like(strips)
packages/core/legoesm/parallel/cubesphere_exchange.py-1345-        for r in range(n_rounds):
packages/core/legoesm/parallel/cubesphere_exchange.py-1346-            send_buf = strips[send_edge_j[r, my_id]]
packages/core/legoesm/parallel/cubesphere_exchange.py:1347:            received = jax.lax.ppermute(send_buf, AXES, tables.perms[r])
packages/core/legoesm/parallel/cubesphere_exchange.py-1348-            tgt = recv_tgt_j[r, my_id][None]
packages/core/legoesm/parallel/cubesphere_exchange.py-1349-            halo_strips = halo_strips.at[tgt].set(
packages/core/legoesm/parallel/cubesphere_exchange.py-1350-                received[None], mode="drop")
--
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
--
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
--
packages/core/legoesm/parallel/cubesphere_exchange.py-1464-        sliv_sn = (sl_if[0], sl_if[1], sl_ib[0], sl_ib[1])
--
packages/core/legoesm/parallel/cubesphere_exchange.py-1704-
packages/core/legoesm/parallel/cubesphere_exchange.py-1705-        # ---- cross-shard ppermute rounds ----
packages/core/legoesm/parallel/cubesphere_exchange.py-1706-        for r in range(n_rounds):
packages/core/legoesm/parallel/cubesphere_exchange.py-1707-            send_buf = my_strips[send_lf_j[r, my_idx],
packages/core/legoesm/parallel/cubesphere_exchange.py-1708-                                 send_le_j[r, my_idx]]  # (max_slots, ...)
packages/core/legoesm/parallel/cubesphere_exchange.py:1709:            received = jax.lax.ppermute(send_buf, "face", perms[r])
packages/core/legoesm/parallel/cubesphere_exchange.py-1710-            tgt = recv_tgt_j[r, my_idx]  # (max_slots,) — 4k sentinel drops
packages/core/legoesm/parallel/cubesphere_exchange.py-1711-            halo_flat = halo_flat.at[tgt].set(received, mode="drop")
packages/core/legoesm/parallel/cubesphere_exchange.py-1712-
--
packages/core/legoesm/parallel/cubesphere_exchange.py-1944-            _cache[key] = _make_exchange_allgather(
packages/core/legoesm/parallel/cubesphere_exchange.py-1945-                mesh, ndim, with_offsets=with_offsets,
packages/core/legoesm/parallel/cubesphere_exchange.py-1946-            )
packages/core/legoesm/parallel/cubesphere_exchange.py-1947-        elif variant == "ppermute_oneface":
packages/core/legoesm/parallel/cubesphere_exchange.py:1948:            _cache[key] = _make_exchange_ppermute(
packages/core/legoesm/parallel/cubesphere_exchange.py-1949-                mesh, ndim, with_offsets=with_offsets,
packages/core/legoesm/parallel/cubesphere_exchange.py-1950-            )
packages/core/legoesm/parallel/cubesphere_exchange.py-1951-        else:  # ppermute_multiface
packages/core/legoesm/parallel/cubesphere_exchange.py-1952-            _cache[key] = _make_exchange_ppermute_multiface(
packages/core/legoesm/parallel/cubesphere_exchange.py-1953-                mesh, ndim, halo=halo, with_offsets=with_offsets,
--
packages/core/legoesm/parallel/async_halo.py-209-        strips = jax.vmap(lambda f: extract_edge_strip(data, f, edge))(
--
packages/core/legoesm/parallel/async_halo.py-211-        )  # (6, n)
packages/core/legoesm/parallel/async_halo.py-212-
packages/core/legoesm/parallel/async_halo.py-213-        # Permute strips between devices.
packages/core/legoesm/parallel/async_halo.py:214:        strips_permuted = jax.lax.ppermute(
packages/core/legoesm/parallel/async_halo.py-215-            strips, axis_name="face", perm=perm,
packages/core/legoesm/parallel/async_halo.py-216-        )
packages/core/legoesm/parallel/async_halo.py-217-
--
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
--
packages/core/legoesm/parallel/sharded_dynamics.py-1904-        send_e = u_shard[se[0]]               # (he_r, nlev)
packages/core/legoesm/parallel/sharded_dynamics.py-1905-        send_c_flat = send_c.ravel()
packages/core/legoesm/parallel/sharded_dynamics.py-1906-        send_packed = jnp.concatenate([send_c_flat, send_e.ravel()])
packages/core/legoesm/parallel/sharded_dynamics.py:1907:        recv_packed = jax.lax.ppermute(
packages/core/legoesm/parallel/sharded_dynamics.py-1908-            send_packed, "device", perm=ppermute_perms[r])
packages/core/legoesm/parallel/sharded_dynamics.py-1909-        split_at = send_c_flat.shape[0]       # static
packages/core/legoesm/parallel/sharded_dynamics.py-1910-        recv_c = recv_packed[:split_at].reshape(send_c.shape)
packages/core/legoesm/parallel/sharded_dynamics.py-1911-        recv_e = recv_packed[split_at:].reshape(send_e.shape)
packages/core/legoesm/parallel/sharded_dynamics.py-1912-        cell_local = cell_local.at[rc[0]].set(recv_c)
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-677-        # band (no r+1) receives the pole-wall 0 via the ppermute non-target.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-678-        def _reconstruct_v(field):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-679-            if field is None:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-680-                return None
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-681-            v_lower = field.data                      # (nl, n_lon[, nlev])
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:682:            boundary = jax.lax.ppermute(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-683-                v_lower[0:1], axis, perm_north)        # band r+1's first row -> 0 at top
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-684-            v_band = jnp.concatenate([v_lower, boundary], axis=0)  # nl+1 rows
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-685-            return field.replace(data=v_band)
--
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2527-        v = x.reshape((1,))
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2528-        acc = v
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2529-        for perm in perms:
packages/core/legoesm/parallel/tiled_production_cdgrid.py:2530:            acc = acc + jax.lax.ppermute(v, AXES, perm)
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2531-        return acc.reshape((1, 1, 1))
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2532-
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2533-    @partial(shard_map, mesh=mesh, in_specs=P(*AXES), out_specs=P(*AXES),
--
packages/core/legoesm/parallel/latlon_spmd.py-107-    perm_to_west, perm_to_east = latlon_lon_ring_perms(p_lon)
packages/core/legoesm/parallel/latlon_spmd.py-108-    # My EAST ghost = east neighbour's west edge (sources send WEST edges to
packages/core/legoesm/parallel/latlon_spmd.py-109-    # their west neighbour); my WEST ghost = west neighbour's east edge.
packages/core/legoesm/parallel/latlon_spmd.py:110:    east_ghost = jax.lax.ppermute(f[:, :halo], "lon", perm_to_west)
packages/core/legoesm/parallel/latlon_spmd.py:111:    west_ghost = jax.lax.ppermute(f[:, -halo:], "lon", perm_to_east)
packages/core/legoesm/parallel/latlon_spmd.py-112-    return jnp.concatenate([west_ghost, f, east_ghost], axis=1)
packages/core/legoesm/parallel/latlon_spmd.py-113-
packages/core/legoesm/parallel/latlon_spmd.py-114-
--
packages/core/legoesm/parallel/latlon_spmd.py-121-    sharded directly; it is carried as ``v_lower = v[:n_lat]`` (``n_lat`` rows,
packages/core/legoesm/parallel/latlon_spmd.py-122-    divisible by ``N``). Each band's NORTH boundary face is the next band's
packages/core/legoesm/parallel/latlon_spmd.py-123-    ``v_lower[0]`` (= the shared global interface row), lifted down via
packages/core/legoesm/parallel/latlon_spmd.py:124:    ``ppermute(..., perm_north)``; the north-most band has no neighbour there and
packages/core/legoesm/parallel/latlon_spmd.py-125-    receives the pole-wall zero (the ppermute non-target). Pure array core (no
packages/core/legoesm/parallel/latlon_spmd.py-126-    Field/state coupling) shared by the ocean and atmosphere lat-band SPMD steps
packages/core/legoesm/parallel/latlon_spmd.py-127-    so the v-stagger numerics are written ONCE (factored from the ocean step's
packages/core/legoesm/parallel/latlon_spmd.py-128-    ``_reconstruct_v`` closure). AD-safe: ``ppermute`` is self-transposing.
packages/core/legoesm/parallel/latlon_spmd.py-129-
--
packages/core/legoesm/parallel/latlon_spmd.py-135-
--
packages/core/legoesm/parallel/latlon_spmd.py-137-    -------
packages/core/legoesm/parallel/latlon_spmd.py-138-    array ``(n_lat_band + 1, n_lon[, nlev])`` — the band's full v-faces.
packages/core/legoesm/parallel/latlon_spmd.py-139-    """
packages/core/legoesm/parallel/latlon_spmd.py:140:    boundary = jax.lax.ppermute(v_lower[0:1], axis, perm_north)
packages/core/legoesm/parallel/latlon_spmd.py-141-    return jnp.concatenate([v_lower, boundary], axis=0)
packages/core/legoesm/parallel/latlon_spmd.py-142-
packages/core/legoesm/parallel/latlon_spmd.py-143-
--
packages/core/legoesm/parallel/latlon_spmd.py-198-            widths.append(w)
packages/core/legoesm/parallel/latlon_spmd.py-199-            flats.append(row.reshape(1, w))
packages/core/legoesm/parallel/latlon_spmd.py-200-        buf = jnp.concatenate(flats, axis=1)
packages/core/legoesm/parallel/latlon_spmd.py-201-        # ONE ppermute for the whole dtype group (north band receives 0).
packages/core/legoesm/parallel/latlon_spmd.py:202:        recv = jax.lax.ppermute(buf, axis, perm_north)
packages/core/legoesm/parallel/latlon_spmd.py-203-        off = 0
packages/core/legoesm/parallel/latlon_spmd.py-204-        for k, i in enumerate(idxs):
packages/core/legoesm/parallel/latlon_spmd.py-205-            w = widths[k]
--
packages/core/legoesm/parallel/latlon_spmd.py-249-        boundary = u_left[:, 0:1]
packages/core/legoesm/parallel/latlon_spmd.py-250-    else:
packages/core/legoesm/parallel/latlon_spmd.py-251-        perm_to_west, _ = latlon_lon_ring_perms(p_lon)
packages/core/legoesm/parallel/latlon_spmd.py:252:        boundary = jax.lax.ppermute(u_left[:, 0:1], axis, perm_to_west)
packages/core/legoesm/parallel/latlon_spmd.py-253-    return jnp.concatenate([u_left, boundary], axis=1)
packages/core/legoesm/parallel/latlon_spmd.py-254-
packages/core/legoesm/parallel/latlon_spmd.py-255-
packages/core/legoesm/parallel/latlon_spmd.py-256-def to_uface_left(u_full):
packages/core/legoesm/parallel/latlon_spmd.py-257-    """Inverse of :func:`reconstruct_uface_left`: drop the east seam column
--
packages/core/legoesm/parallel/latlon_spmd.py-387-    # 1. E/W-extend the edge rows by 2h (one lon ring ppermute pair).
packages/core/legoesm/parallel/latlon_spmd.py-388-    ext = lon_ring_ghosts_spmd(edge, mesh, halo=2 * h)  # (halo, w+4h[, lev])
packages/core/legoesm/parallel/latlon_spmd.py-389-    # 2. ONE antipodal ppermute over the lon ring (shift by p_lon/2 is a
packages/core/legoesm/parallel/latlon_spmd.py-390-    # bijection, and c' != c for every even p_lon >= 2).
packages/core/legoesm/parallel/latlon_spmd.py-391-    perm_anti = [(s, (s + p_lon // 2) % p_lon) for s in range(p_lon)]
packages/core/legoesm/parallel/latlon_spmd.py:392:    recv = jax.lax.ppermute(ext, "lon", perm_anti)
packages/core/legoesm/parallel/latlon_spmd.py-393-    # 3. lat-mirror + sign (the _pole_fold row flip), then the window
packages/core/legoesm/parallel/latlon_spmd.py-394-    # column map (derivation above; seam-straddling windows mix branches).
packages/core/legoesm/parallel/latlon_spmd.py-395-    sign = -1.0 if negate else 1.0
--
packages/core/legoesm/parallel/latlon_spmd.py-397-    t = jnp.arange(w + 2 * h)
--
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
--
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
--
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
--
packages/core/legoesm/parallel/latlon_spmd.py-705-                              fields[i][-halo:].reshape(halo, w)))
packages/core/legoesm/parallel/latlon_spmd.py-706-            south_buf = jnp.concatenate([s for s, _ in flats], axis=1)
packages/core/legoesm/parallel/latlon_spmd.py-707-            north_buf = jnp.concatenate([n for _, n in flats], axis=1)
packages/core/legoesm/parallel/latlon_spmd.py-708-            # ONE ppermute pair for the whole dtype group.
packages/core/legoesm/parallel/latlon_spmd.py:709:            north_recv = jax.lax.ppermute(south_buf, axis, perm_north)
packages/core/legoesm/parallel/latlon_spmd.py:710:            south_recv = jax.lax.ppermute(north_buf, axis, perm_south)
packages/core/legoesm/parallel/latlon_spmd.py-711-            off = 0
packages/core/legoesm/parallel/latlon_spmd.py-712-            for k, i in enumerate(idxs):
packages/core/legoesm/parallel/latlon_spmd.py-713-                w = widths[k]

exec
/usr/bin/bash -lc "rg -n -C 20 \"def make_sharded_atm_latlon_step|return band_step|return.*step|def make_sharded_atm_latlon_segment|unroll_to_dtype_fixed_point\" packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py
sed -n '620,845p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py
sed -n '180,285p' packages/core/legoesm/core/precision.py
rg -n \"set_policy\\(|LEG.*PREC|PRECISION|precision\" scripts/cluster/scaling_levante/_env.sh scripts/bench/bench_atm_latlon_spmd_scaling.py packages/core/legoesm/core/precision.py | sed -n '1,260p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 226ms:
415-        gi = 0 if shard_geometry else jax.lax.axis_index(axis)
416-        band_geom = template._replace(
417-            **{name: stacks_local[name][gi] for name in array_field_names})
418-        pmask = (stacks_local["__polar_mask"][gi]
419-                 if "__polar_mask" in stacks_local else None)
420-        pmaskv = (stacks_local["__polar_mask_v"][gi]
421-                  if "__polar_mask_v" in stacks_local else None)
422-        # Reconstruct the band's nl+1 v-faces from v_lower (shared interface
423-        # row via ppermute), run the un-jitted band step, convert v back.
424-        v_full = reconstruct_vface_lower(state_local.v, axis, perm_north)
425-        state_band = state_local._replace(v=v_full)
426-        out, ps_out = model._step_cgrid_impl(
427-            state_band, dt,
428-            physics_fn=physics_fn, phys_state=ps_local,
429-            grid=band_geom, sigma_coord=model.sigma_coord,
430-            polar_mask=pmask, polar_mask_v=pmaskv,
431-            pole_v_bc_masks=spmd_pole_end_masks(),
432-        )
433-        return out._replace(v=to_vface_lower(out.v)), ps_out
434-
435:    return band_step
436-
437-
438-def _dtype_sig(tree) -> tuple:
439-    """Trace-time carry signature of a pytree: the pytree STRUCTURE plus
440-    every array leaf's ``(shape, dtype, weak_type)``.
441-
442-    ``lax.scan`` requires carry-in == carry-out in ALL of structure, shape,
443-    dtype and weak type — dtype strings alone would mark a step that flips
444-    weak typing (or shape) while preserving dtypes as "stable" and then
445-    fail inside the scan lowering instead of being absorbed by the unroll
446-    (codex M3b review).  Non-array leaves contribute through the treedef.
447-    """
448-    leaves, treedef = jax.tree_util.tree_flatten(tree)
449-    return (str(treedef),
450-            tuple((tuple(leaf.shape), str(leaf.dtype),
451-                   bool(getattr(leaf, "weak_type", False)))
452-                  for leaf in leaves if hasattr(leaf, "dtype")))
453-
454-
455:def unroll_to_dtype_fixed_point(step1, state, n_left: int):
456-    """Unroll ``step1`` applications until the state's dtype signature is a
457-    FIXED POINT of the step (bounded by ``n_left``); returns
458-    ``(state, n_left_remaining)``.
459-
460-    PUBLIC (M3b): shared by this module's lat-band segments and the tiled
461-    cube segment (``tiled_step_adapter.scan_tiled_cc_steps``) — any
462-    ``lax.scan``-of-a-step needs carry-in == carry-out dtypes, and this is
463-    the ONE place that trace-time unroll lives (no re-derivation).
464-
465-    Why: ``lax.scan`` needs carry-in == carry-out dtypes.  A mixed-precision
466-    IC (f32 grid/init-derived leaves beside the step's f64 sources under
467-    x64 — the f64 sigma-coordinate arrays and the strong-f64
468-    ``jnp.asarray(dt)`` operand) promotes over the first stepS exactly as
469-    the per-step Python loop absorbs silently: observed on the bench IC
470-    (jobs 8916406/8916740), ``p_s`` promotes f32->f64 in step 1 and
471-    ``u/v/T`` follow in step 2 by mixing the now-f64 ``p_s`` — the fixed
472-    point can take MORE than one application, and no role-based cast can
473-    express it (the default f32 policy no-ops while the promotion is
474-    mixed-LEAF arithmetic).
475-
--
513-    mesh) after — a later serial/full-domain call must not take SPMD-only
514-    branches outside a shard_map.  The first call through a jitted fn traces
515-    with the backend armed (baking the band halo); later calls reuse the
516-    cached compile and the arm/restore keeps interleaved serial paths
517-    untouched."""
518-    from legoesm.grids.halo import (
519-        get_halo_backend, get_mpi_topology, get_spmd_mesh,
520-        set_halo_backend, set_spmd_mesh)
521-    from legoesm.parallel.latlon_spmd import activate_latlon_spmd_halo
522-    prev_backend = get_halo_backend()
523-    prev_topo = get_mpi_topology()
524-    prev_mesh = get_spmd_mesh()
525-    activate_latlon_spmd_halo(mesh)
526-    try:
527-        yield
528-    finally:
529-        set_spmd_mesh(prev_mesh)
530-        set_halo_backend(prev_backend, prev_topo)
531-
532-
533:def make_sharded_atm_latlon_step(model, mesh, physics_fn=None, *,
534-                                 shard_geometry: bool = False):
535-    """Return ``step(c_state, dt) -> c_state`` running the C-grid hydrostatic atm
536-    step lat-band-SPMD over the 1-D ``"lat"`` mesh.
537-
538-    ``c_state`` is a ``CGridLatLonHydrostaticState`` laid out with
539-    :func:`shard_state_atm_latlon` (``v`` carried as the ``n_lat``-row
540-    ``v_lower``). The body reconstructs each band's ``nl+1`` v-faces, runs the
541-    UN-jitted ``model._step_cgrid_impl`` on the band geometry + band polar masks
542-    + per-band pole masks, and converts the result ``v`` back to ``v_lower``.
543-    REUSES the shared primitives (band perms, the v-face round-trip, the band
544-    halo via the swapped backend, ``spmd_pole_end_masks``, the band slicer);
545-    atm-NEW is only the 6-field state/geometry walk. ``check_vma=False`` (the
546-    band halo reads neighbour-rank data). Mirrors ``make_sharded_ocean_step``.
547-
548-    ``physics_fn`` (optional): a STATELESS, COLUMN-LOCAL physics closure
549-    (``physics_fn(hs, grid, sigma_coord) -> HydrostaticTendencies``, e.g.
550-    Held-Suarez or any per-column parameterization). It is evaluated inside each
551-    RK stage on the BAND geometry (``_step_cgrid_impl`` routes the band grid into
552-    ``_call_physics`` so a lat-dependent forcing sees the band's latitudes), and
553-    its wind tendencies couple cell->face through the SPMD-aware
--
573-    geometry slice instead of a replicated all-band copy (1/n_dev the
574-    bytes, :func:`atm_latlon_geometry_bytes`).  The body consumes the SAME
575-    band values either way, so the step is bit-identical (gated by
576-    ``tests/parallel/test_atm_latlon_segment.py``).
577-    """
578-    from legoesm.parallel.latlon_spmd import latlon_band_perms
579-    from legoesm.parallel.shard_map_compat import shard_map
580-
581-    # Stochastic physics is SPMD-safe since increment 2: the Bechtold AR1
582-    # innovation folds the per-step sub-key with each column's GLOBAL id
583-    # (``PhysicsState.col_index`` — band-split with the carry, so every
584-    # shard holds its own global ids) and the replicated master key splits
585-    # identically on every band — the draw is decomposition-INVARIANT.
586-    # Deterministic prognostic carries (tke/qke, conv profiles, GWD
587-    # spectrum) thread exactly: the ColumnAdapter flatten is a C-order
588-    # (lat-major) reshape, so a contiguous dim-0 shard of every
589-    # ``(ncol, ...)`` PhysicsState leaf IS the band's own columns.
590-
591-    # A stateful (tagged) physics_fn with NO carry would silently reseed
592-    # its PhysicsState every step (issue #405/#413) — model.step()'s
593:    # guard is bypassed here, so the returned step re-checks per call.
594-    from legoesm.timestepping.integration import (
595-        refuse_unthreaded_stateful_physics)
596-
597-    if mesh is None:                       # single-device: plain C-grid step
598-        def _serial_step(c_state, dt, phys_state=None):
599-            refuse_unthreaded_stateful_physics(
600-                physics_fn, phys_state, where="atm lat-band SPMD step")
601-            out, ps_out = model._step_cgrid(
602-                c_state, dt, physics_fn=physics_fn, phys_state=phys_state)
603-            return out if phys_state is None else (out, ps_out)
604:        return _serial_step
605-
606-    n_dev = mesh.devices.size
607-    axis = mesh.axis_names[0]
608-    grid = model.grid
609-
610-    _refuse_unsupported_spmd_config(model)
611-
612-    template, array_field_names, stacks, stacks_spec = _build_geometry_stacks(
613-        model, mesh, n_dev, shard_geometry)
614-
615-    perm_north, _perm_south = latlon_band_perms(n_dev)
616-    band_step = _make_band_step_body(
617-        model, template, array_field_names, axis, perm_north, physics_fn,
618-        shard_geometry)
619-
620-    def _body(state_local, stacks_local, dt):
621-        out, _ = band_step(state_local, stacks_local, dt, None)
622-        return out
623-
624-    def _body_with_carry(state_local, stacks_local, dt, ps_local):
625-        # Stateful variant: the band's PhysicsState chunk (ncol_band =
626-        # nl*n_lon leading dim — the C-order lat-major flatten makes a
627-        # contiguous dim-0 shard exactly the band's own columns) is
628-        # threaded into every RK stage and the carry-out returned.
629:        return band_step(state_local, stacks_local, dt, ps_local)
630-
631-    _ncol_global = int(grid.n_lat) * int(grid.n_lon)
632-
633-    def _ps_spec_leaf(leaf):
634-        # (ncol, ...) leaves band-split on dim 0 (lat-major flatten);
635-        # everything else (prng_key (2,), scalars) replicated.
636-        if (hasattr(leaf, "ndim") and leaf.ndim >= 1
637-                and leaf.shape[0] == _ncol_global):
638-            return P("lat")
639-        return P()
640-
641-    # Build the JITTED shard_map ONCE and cache it. ``jax.jit`` is LOAD-BEARING:
642-    # a bare shard_map is NOT compilation-cached, so calling it re-traces +
643-    # recompiles the (large, un-jitted) band step EVERY call — a 32x64x10 nd=2
644-    # step took ~142 s/step (bench 8560671), and the whole equivalence gate ran
645-    # ~65 min. Wrapping in jit caches the compile: probe 8561202 measured
646-    # [3079, 1.4, 1.2, 1.1, 1.1] ms — first call compiles, the rest hit the
647-    # cache. ``dt`` is a TRACED operand (not a closure constant) so a changing dt
648-    # does not retrigger compilation. The grid-tracer concern that kept
649-    # _step_cgrid_impl un-jitted does NOT bite here: band_geom's STATIC scalar
--
668-        if fn is None:
669-            in_spec = jax.tree.map(lat_spec, c_state)
670-            if phys_state is None:
671-                fn = jax.jit(shard_map(
672-                    _body, mesh=mesh, in_specs=(in_spec, stacks_spec, P()),
673-                    out_specs=in_spec, check_vma=False))
674-            else:
675-                ps_spec = jax.tree.map(_ps_spec_leaf, phys_state)
676-                fn = jax.jit(shard_map(
677-                    _body_with_carry, mesh=mesh,
678-                    in_specs=(in_spec, stacks_spec, P(), ps_spec),
679-                    out_specs=(in_spec, ps_spec), check_vma=False))
680-            _cache[key] = fn
681-        # Arm the SPMD band halo around the call ONLY (see _latlon_spmd_armed).
682-        with _latlon_spmd_armed(mesh):
683-            if phys_state is None:
684-                return fn(c_state, stacks, jnp.asarray(dt))
685-            return fn(c_state, stacks, jnp.asarray(dt), phys_state)
686-
687-    sharded_step._geom_stacks = stacks   # test/introspection only
688:    return sharded_step
689-
690-
691:def make_sharded_atm_latlon_segment(model, mesh, n_steps: int,
692-                                    physics_fn=None, *,
693-                                    shard_geometry: bool = True):
694-    """Return ``segment(c_state, dt) -> (c_state, all_finite)`` advancing
695-    ``n_steps`` C-grid steps in ONE compiled program — a ``lax.scan`` of the
696-    band step inside a single jitted ``shard_map``, built once and reused
697-    (the M2b lever: "compile atmosphere lat-lon segments instead of
698-    launching one step at a time").
699-
700-    Contrast with driving :func:`make_sharded_atm_latlon_step` in a Python
701-    loop: ONE host dispatch (+ halo-backend arm/restore + cache-key hash) per
702-    SEGMENT instead of per STEP, and no per-step host round-trip between
703-    device launches.  The scanned band body is the SAME
704-    ``_make_band_step_body`` the per-step path runs, so the trajectory
705-    matches the sequential sharded steps to compilation-order roundoff
706-    (gated at 1e-12 by ``tests/parallel/test_atm_latlon_segment.py``).
707-
708-    ``all_finite`` is a REPLICATED traced scalar bool from
709-    :func:`state_finite_scalar` — the in-graph blowup guard (``psum`` of
710-    per-band non-finite presence over ALL state leaves).  The host reads
711-    this ONE scalar per segment instead of gathering the full state.
--
715-    (``P("lat")`` stacks) instead of a replicated all-band copy —
716-    bit-identical numerics, 1/n_dev the geometry bytes
717-    (:func:`atm_latlon_geometry_bytes`).
718-
719-    STATELESS physics only (``None`` / Held-Suarez / column-local closures,
720-    the production ``run_atm_latlon_spmd`` envelope): a stateful
721-    ``PhysicsState`` carry is refused loudly — thread it through the
722-    per-step :func:`make_sharded_atm_latlon_step` until the segment lane
723-    routes the carry through the scan.
724-
725-    ``mesh=None``: the single-device twin — ``jit(lax.scan)`` over the serial
726-    C-grid step with the model's own geometry, same ``(state, all_finite)``
727-    contract.
728-
729-    ``n_steps`` is STATIC (the compiled scan length): one compiled program
730-    per distinct segment length (``run_atm_latlon_spmd`` caches per length —
731-    at most two: the regular segment and the final remainder).
732-
733-    Carry dtype: leading steps are UNROLLED outside the ``lax.scan`` until
734-    the state's dtype signature is a fixed point of the step
735:    (:func:`unroll_to_dtype_fixed_point` — ``jax.eval_shape`` probe, zero
736-    FLOPs, trace-time constant).  A mixed-precision IC promotes over the
737-    first stepS (``p_s`` first, ``u/v/T`` next via the promoted ``p_s`` —
738-    observed jobs 8916406/8916740) exactly as the per-step Python loop
739-    absorbs silently; an already-stable state unrolls NOTHING and scans all
740-    ``n_steps``.  Zero extra casts, zero numerical difference vs the
741-    per-step lane — never a silent precision change.
742-    """
743-    from legoesm.parallel.latlon_spmd import latlon_band_perms
744-    from legoesm.parallel.shard_map_compat import shard_map
745-    from legoesm.timestepping.integration import (
746-        refuse_unthreaded_stateful_physics)
747-
748-    if int(n_steps) < 1:
749-        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
750-    n_steps = int(n_steps)
751-
752-    def _refuse_carry(phys_state):
753-        refuse_unthreaded_stateful_physics(
754-            physics_fn, phys_state, where="atm lat-lon compiled segment")
755-        if phys_state is not None:
756-            raise NotImplementedError(
757-                "make_sharded_atm_latlon_segment: a stateful PhysicsState "
758-                "carry is not yet segment-routed — use the per-step "
759-                "make_sharded_atm_latlon_step(phys_state=...) path.")
760-
761-    if mesh is None:                       # single-device compiled segment
762-        def _serial_seg(c_state, dt):
763-            def _step1(s):
764-                out, _ps = model._step_cgrid_impl(
765-                    s, dt, physics_fn=physics_fn, phys_state=None)
766-                return out
767-            # Unroll to the scan-carry dtype fixed point (helper docstring).
768:            out, n_left = unroll_to_dtype_fixed_point(
769-                _step1, c_state, n_steps)
770-            if n_left > 0:
771-                out, _ = jax.lax.scan(lambda s, _x: (_step1(s), None),
772-                                      out, xs=None, length=n_left)
773-            return out, state_finite_scalar(out)
774-
775-        fn_serial = jax.jit(_serial_seg)
776-
777-        def serial_segment(c_state, dt, phys_state=None):
778-            _refuse_carry(phys_state)
779-            return fn_serial(c_state, jnp.asarray(dt))
780-
781-        return serial_segment
782-
783-    n_dev = mesh.devices.size
784-    axis = mesh.axis_names[0]
785-
786-    _refuse_unsupported_spmd_config(model)
787-
788-    template, array_field_names, stacks, stacks_spec = _build_geometry_stacks(
789-        model, mesh, n_dev, shard_geometry)
790-    perm_north, _perm_south = latlon_band_perms(n_dev)
791-    band_step = _make_band_step_body(
792-        model, template, array_field_names, axis, perm_north, physics_fn,
793-        shard_geometry)
794-
795-    def _seg_body(state_local, stacks_local, dt):
796-        def _step1(s):
797-            out, _ps = band_step(s, stacks_local, dt, None)
798-            return out
799-        # Unroll to the scan-carry dtype fixed point (helper docstring).
800:        out, n_left = unroll_to_dtype_fixed_point(
801-            _step1, state_local, n_steps)
802-        if n_left > 0:
803-            out, _ = jax.lax.scan(lambda s, _x: (_step1(s), None),
804-                                  out, xs=None, length=n_left)
805-        return out, state_finite_scalar(out, axis=axis)
806-
807-    _cache = {}
808-
809-    def segment(c_state, dt, phys_state=None):
810-        _refuse_carry(phys_state)
811-        # Cache key = state pytree STRUCTURE (in/out specs derive from it) —
812-        # same doctrine as the per-step factory.
813-        key = jax.tree.structure(c_state)
814-        fn = _cache.get(key)
815-        if fn is None:
816-            in_spec = jax.tree.map(lat_spec, c_state)
817-            fn = jax.jit(shard_map(
818-                _seg_body, mesh=mesh,
819-                in_specs=(in_spec, stacks_spec, P()),
820-                out_specs=(in_spec, P()), check_vma=False))
--
966-
967-        def _seg_fn(n):
968-            fn = seg_fns.get(n)
969-            if fn is None:
970-                fn = make_sharded_atm_latlon_segment(
971-                    model, mesh, n, physics_fn=physics_fn)
972-                seg_fns[n] = fn
973-            return fn
974-
975-        c_state = shard_hydrostatic_to_atm_latlon(hs_init, model.grid, mesh)
976-        done = 0
977-        while done < n_steps:
978-            this = min(seg, n_steps - done)
979-            c_state, ok = _seg_fn(this)(c_state, dt)
980-            done += this
981-            # ONE scalar host read per segment (the M2b finite check); the
982-            # full state is gathered only on blowup / for the callback.
983-            if not bool(ok):
984-                hs_out = gather_atm_latlon_to_hydrostatic(
985-                    c_state, model.grid, mesh)
986:                return hs_out, f"BLOWUP at step {done}"
987-            if on_segment is not None:
988-                on_segment(gather_atm_latlon_to_hydrostatic(
989-                    c_state, model.grid, mesh), done)
990-        return (gather_atm_latlon_to_hydrostatic(c_state, model.grid, mesh),
991-                "COMPLETED")
992-
993-    step = make_sharded_atm_latlon_step(model, mesh, physics_fn=physics_fn)
994-    # Convert + shard ONCE; the run stays in the sharded C-grid layout.
995-    c_state = shard_hydrostatic_to_atm_latlon(hs_init, model.grid, mesh)
996-    done = 0
997-    status = "COMPLETED"
998-    while done < n_steps:
999-        this = min(seg, n_steps - done)
1000-        for _ in range(this):
1001-            c_state = step(c_state, dt)
1002-        done += this
1003-        # Gather a cell-centered COPY for output / blowup-check ONLY — the
1004-        # integration continues from c_state (sharded C-grid), so the lossy
1005-        # cell<->face round-trip is NOT fed back into the dynamics.
1006-        hs_out = gather_atm_latlon_to_hydrostatic(c_state, model.grid, mesh)
1007-        finite = bool(jnp.isfinite(hs_out.p_s.data).all()
1008-                      & jnp.isfinite(hs_out.T.data).all())
1009-        if not finite:
1010:            return hs_out, f"BLOWUP at step {done}"
1011-        if on_segment is not None:
1012-            on_segment(hs_out, done)
1013-    return gather_atm_latlon_to_hydrostatic(c_state, model.grid, mesh), status
1014-
1015-
1016-# ==============================================================================
1017-# M3a — native 2-D ("lat", "lon") tiling for the atmosphere SPMD step
1018-# ==============================================================================
1019-# The 1-D lat-band decomposition's halo perimeter is the CONSTANT n_lon per
1020-# cut (independent of the device count) — the term that caps band scaling.
1021-# The 2-D tiling shards latitude AND longitude: lat stays the pole-terminated
1022-# line (ppermute at cuts, the serial 180-deg fold at the pole tiles), lon
1023-# becomes a periodic ring (cyclic ppermute — the wrap IS the roll
1024-# permutation).  Staggered ownership mirrors the 1-D v convention:
1025-#
1026-#   * v (n_lat+1 rows)   -> v_lower = v[:n_lat]; each tile's north boundary
1027-#     face is the lat-neighbour's v_lower[0] (reconstruct_vface_lower).
1028-#   * u (n_lon+1 columns) -> u_left = u[:, :n_lon]; each tile's east seam
1029-#     face is the lon-neighbour's u_left[:, 0] (reconstruct_uface_left) —
1030-#     the +1 seam column is OWNED by the tile whose slice starts there and
--
1240-        pmask = (stacks_local["__polar_mask"][gi, gj]
1241-                 if "__polar_mask" in stacks_local else None)
1242-        pmaskv = (stacks_local["__polar_mask_v"][gi, gj]
1243-                  if "__polar_mask_v" in stacks_local else None)
1244-        # Reconstruct the tile's nl+1 v-faces (shared interface row via the
1245-        # lat ppermute) and w+1 u-faces (periodic seam column via the lon
1246-        # ring), run the un-jitted step, convert both staggers back.
1247-        v_full = reconstruct_vface_lower(state_local.v, "lat", perm_north)
1248-        u_full = reconstruct_uface_left(state_local.u, "lon", p_lon)
1249-        state_tile = state_local._replace(u=u_full, v=v_full)
1250-        out, ps_out = model._step_cgrid_impl(
1251-            state_tile, dt,
1252-            physics_fn=physics_fn, phys_state=ps_local,
1253-            grid=tile_geom, sigma_coord=model.sigma_coord,
1254-            polar_mask=pmask, polar_mask_v=pmaskv,
1255-            pole_v_bc_masks=spmd_pole_end_masks(),
1256-        )
1257-        return (out._replace(u=to_uface_left(out.u),
1258-                             v=to_vface_lower(out.v)), ps_out)
1259-
1260:    return tile_step
1261-
1262-
1263-def _check_2d_mesh(mesh) -> tuple[int, int]:
1264-    """Validate the 2-D tile mesh axes and return ``(p_lat, p_lon)``."""
1265-    names = tuple(mesh.axis_names)
1266-    if names != ("lat", "lon"):
1267-        raise ValueError(
1268-            f"atm 2-D SPMD tiling: mesh axes must be ('lat', 'lon'); got "
1269-            f"{names}.  Build it as Mesh(devices.reshape(p_lat, p_lon), "
1270-            f"axis_names=('lat', 'lon')) — choose_latlon_2d_topology picks "
1271-            f"(p_lat, p_lon).")
1272-    return int(mesh.shape["lat"]), int(mesh.shape["lon"])
1273-
1274-
1275-def _refuse_unsupported_spmd_config_2d(model, p_lon: int) -> None:
1276-    """2-D-specific dispatch-hardening on top of the shared band refusals."""
1277-    _refuse_unsupported_spmd_config(model)
1278-    if p_lon > 1 and bool(getattr(model.config, "use_polar_filter", False)):
1279-        raise NotImplementedError(
1280-            "atm 2-D SPMD tiling: use_polar_filter=True with p_lon > 1 is "
1281-            "not wired — the polar filter FFTs the full longitude circle "
1282-            "and needs a lon-gather FFT.  (The route-A MPI path "
1283-            "make_latlon_2d_mpi_step DOES wire this via the AD-safe "
1284-            "lat-pencil transpose; the SPMD ppermute equivalent is a "
1285-            "follow-up.)  Use p_lon == 1 or disable the filter.")
1286-
1287-
1288:def make_sharded_atm_latlon_step_2d(model, mesh, physics_fn=None, *,
1289-                                    shard_geometry: bool = True):
1290-    """Return ``step(c_state, dt) -> c_state`` running the C-grid hydrostatic
1291-    atm step 2-D-tile-SPMD over a ``("lat", "lon")`` mesh — the M3a native
1292-    2-D tiling twin of :func:`make_sharded_atm_latlon_step`.
1293-
1294-    ``c_state`` is a ``CGridLatLonHydrostaticState`` laid out with
1295-    :func:`shard_state_atm_latlon_2d` (``v`` as ``v_lower``, ``u`` as
1296-    ``u_left``).  The body reconstructs each tile's staggered faces (v via
1297-    the lat ppermute, u via the periodic lon ring), runs the UN-jitted
1298-    ``model._step_cgrid_impl`` on the tile geometry + per-tile pole masks,
1299-    and converts both staggers back.  Halos: the armed 2-D SPMD backend
1300-    routes ``pad_halo_latlon*`` through ``make_latlon_2d_pad_body`` (lat
1301-    ppermute + lon ring + the EXACT serial 180-deg pole fold via a lon-ring
1302-    all_gather at the pole tiles), ``pad_with_pole_bc_lat`` through the
1303-    lat-only wall body, and ``pad_lon_cgrid`` through the lon ring — all
1304-    shared machinery, no operator numerics duplicated.  Global reductions
1305-    (the mass fixer's ``batch_global_area_sums``) psum over BOTH mesh axes.
1306-
1307-    A degenerate ``(N, 1)`` mesh is bit-identical to the 1-D band step
1308-    (every lon-ring op takes its static local branch; gated by
--
1310-    remains the default production lane.
1311-
1312-    ``physics_fn``: STATELESS column-local closures only (Held-Suarez etc.),
1313-    evaluated per RK stage on the TILE geometry — decomposition-invariant
1314-    with no collectives.  A stateful ``PhysicsState`` carry is REFUSED: its
1315-    ``(ncol, ...)`` leaves flatten lat-major over the GLOBAL grid, so a
1316-    contiguous dim-0 shard is a lat BAND's columns, not a 2-D tile's —
1317-    thread carries through the 1-D :func:`make_sharded_atm_latlon_step`.
1318-
1319-    ``shard_geometry=True`` (default — new API, no historical layout):
1320-    per-device tile geometry slices (``P("lat", "lon")`` stacks);
1321-    ``False`` replicates the all-tile stacks (indexed at the axis indices).
1322-    Same tile values either way (bit-identical numerics).
1323-    """
1324-    from legoesm.parallel.latlon_spmd import latlon_band_perms
1325-    from legoesm.parallel.shard_map_compat import shard_map
1326-    from legoesm.timestepping.integration import (
1327-        refuse_unthreaded_stateful_physics)
1328-
1329-    if mesh is None:                       # single-device: plain C-grid step
1330:        return make_sharded_atm_latlon_step(model, None,
1331-                                            physics_fn=physics_fn)
1332-
1333-    p_lat, p_lon = _check_2d_mesh(mesh)
1334-    _refuse_unsupported_spmd_config_2d(model, p_lon)
1335-
1336-    template, array_field_names, stacks, stacks_spec = (
1337-        _build_geometry_stacks_2d(model, mesh, p_lat, p_lon, shard_geometry))
1338-    perm_north, _perm_south = latlon_band_perms(p_lat)
1339-    tile_step = _make_tile_step_body_2d(
1340-        model, template, array_field_names, perm_north, p_lon, physics_fn,
1341-        shard_geometry)
1342-
1343-    def _body(state_local, stacks_local, dt):
1344-        out, _ = tile_step(state_local, stacks_local, dt, None)
1345-        return out
1346-
1347-    _cache = {}
1348-
1349-    def sharded_step(c_state, dt, phys_state=None):
1350-        refuse_unthreaded_stateful_physics(
1351-            physics_fn, phys_state, where="atm lat-lon 2-D SPMD step")
1352-        if phys_state is not None:
1353-            raise NotImplementedError(
1354-                "make_sharded_atm_latlon_step_2d: a stateful PhysicsState "
1355-                "carry is not 2-D-tile-routed (its (ncol, ...) leaves "
1356-                "flatten lat-major over the GLOBAL grid — a contiguous "
1357-                "dim-0 shard is a lat band, not a 2-D tile).  Thread the "
1358-                "carry through the 1-D make_sharded_atm_latlon_step.")
1359-        key = jax.tree.structure(c_state)
1360-        fn = _cache.get(key)
1361-        if fn is None:
1362-            in_spec = jax.tree.map(tile_spec, c_state)
1363-            fn = jax.jit(shard_map(
1364-                _body, mesh=mesh, in_specs=(in_spec, stacks_spec, P()),
1365-                out_specs=in_spec, check_vma=False))
1366-            _cache[key] = fn
1367-        with _latlon_spmd_armed(mesh):
1368-            return fn(c_state, stacks, jnp.asarray(dt))
1369-
1370-    sharded_step._geom_stacks = stacks   # test/introspection only
1371:    return sharded_step
1372-
1373-
1374:def make_sharded_atm_latlon_segment_2d(model, mesh, n_steps: int,
1375-                                       physics_fn=None, *,
1376-                                       shard_geometry: bool = True):
1377-    """Return ``segment(c_state, dt) -> (c_state, all_finite)`` advancing
1378-    ``n_steps`` C-grid steps in ONE compiled ``lax.scan`` over the 2-D
1379-    ``("lat", "lon")`` tile mesh — the M3a twin of
1380-    :func:`make_sharded_atm_latlon_segment` (same contract: static
1381-    ``n_steps``, replicated in-graph finite scalar psum'd over BOTH mesh
1382-    axes, leading steps unrolled to the scan-carry dtype fixed point via
1383:    :func:`unroll_to_dtype_fixed_point`, STATELESS physics only,
1384-    ``mesh=None`` -> the single-device compiled twin)."""
1385-    from legoesm.parallel.latlon_spmd import latlon_band_perms
1386-    from legoesm.parallel.shard_map_compat import shard_map
1387-    from legoesm.timestepping.integration import (
1388-        refuse_unthreaded_stateful_physics)
1389-
1390-    if int(n_steps) < 1:
1391-        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
1392-    n_steps = int(n_steps)
1393-
1394-    if mesh is None:                       # single-device compiled segment
1395:        return make_sharded_atm_latlon_segment(model, None, n_steps,
1396-                                               physics_fn=physics_fn)
1397-
1398-    p_lat, p_lon = _check_2d_mesh(mesh)
1399-    _refuse_unsupported_spmd_config_2d(model, p_lon)
1400-
1401-    def _refuse_carry(phys_state):
1402-        refuse_unthreaded_stateful_physics(
1403-            physics_fn, phys_state, where="atm lat-lon 2-D compiled segment")
1404-        if phys_state is not None:
1405-            raise NotImplementedError(
1406-                "make_sharded_atm_latlon_segment_2d: a stateful PhysicsState "
1407-                "carry is not 2-D-tile-routed — use the 1-D per-step "
1408-                "make_sharded_atm_latlon_step(phys_state=...) path.")
1409-
1410-    template, array_field_names, stacks, stacks_spec = (
1411-        _build_geometry_stacks_2d(model, mesh, p_lat, p_lon, shard_geometry))
1412-    perm_north, _perm_south = latlon_band_perms(p_lat)
1413-    tile_step = _make_tile_step_body_2d(
1414-        model, template, array_field_names, perm_north, p_lon, physics_fn,
1415-        shard_geometry)
1416-
1417-    def _seg_body(state_local, stacks_local, dt):
1418-        def _step1(s):
1419-            out, _ps = tile_step(s, stacks_local, dt, None)
1420-            return out
1421-        # Unroll to the scan-carry dtype fixed point (helper docstring).
1422-        # Public name (the _-prefixed original was promoted; the 2-D path
1423-        # kept the stale private reference — NameError on first segment
1424-        # trace, caught by test_2d_segment_matches_sequential_and_serial).
1425:        out, n_left = unroll_to_dtype_fixed_point(
1426-            _step1, state_local, n_steps)
1427-        if n_left > 0:
1428-            out, _ = jax.lax.scan(lambda s, _x: (_step1(s), None),
1429-                                  out, xs=None, length=n_left)
1430-        return out, state_finite_scalar(out, axis=("lat", "lon"))
1431-
1432-    _cache = {}
1433-
1434-    def segment(c_state, dt, phys_state=None):
1435-        _refuse_carry(phys_state)
1436-        key = jax.tree.structure(c_state)
1437-        fn = _cache.get(key)
1438-        if fn is None:
1439-            in_spec = jax.tree.map(tile_spec, c_state)
1440-            fn = jax.jit(shard_map(
1441-                _seg_body, mesh=mesh,
1442-                in_specs=(in_spec, stacks_spec, P()),
1443-                out_specs=(in_spec, P()), check_vma=False))
1444-            _cache[key] = fn
1445-        with _latlon_spmd_armed(mesh):
    def _body(state_local, stacks_local, dt):
        out, _ = band_step(state_local, stacks_local, dt, None)
        return out

    def _body_with_carry(state_local, stacks_local, dt, ps_local):
        # Stateful variant: the band's PhysicsState chunk (ncol_band =
        # nl*n_lon leading dim — the C-order lat-major flatten makes a
        # contiguous dim-0 shard exactly the band's own columns) is
        # threaded into every RK stage and the carry-out returned.
        return band_step(state_local, stacks_local, dt, ps_local)

    _ncol_global = int(grid.n_lat) * int(grid.n_lon)

    def _ps_spec_leaf(leaf):
        # (ncol, ...) leaves band-split on dim 0 (lat-major flatten);
        # everything else (prng_key (2,), scalars) replicated.
        if (hasattr(leaf, "ndim") and leaf.ndim >= 1
                and leaf.shape[0] == _ncol_global):
            return P("lat")
        return P()

    # Build the JITTED shard_map ONCE and cache it. ``jax.jit`` is LOAD-BEARING:
    # a bare shard_map is NOT compilation-cached, so calling it re-traces +
    # recompiles the (large, un-jitted) band step EVERY call — a 32x64x10 nd=2
    # step took ~142 s/step (bench 8560671), and the whole equivalence gate ran
    # ~65 min. Wrapping in jit caches the compile: probe 8561202 measured
    # [3079, 1.4, 1.2, 1.1, 1.1] ms — first call compiles, the rest hit the
    # cache. ``dt`` is a TRACED operand (not a closure constant) so a changing dt
    # does not retrigger compilation. The grid-tracer concern that kept
    # _step_cgrid_impl un-jitted does NOT bite here: band_geom's STATIC scalar
    # fields stay concrete (template._replace only swaps the array fields), and
    # the SPMD operator retrofits removed the trace-time static-bool checks on
    # the cut/pole branches.
    _cache = {}

    def sharded_step(c_state, dt, phys_state=None):
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="atm lat-band SPMD step")
        # Cache key = (state pytree STRUCTURE, phys_state pytree structure):
        # in_specs/out_specs derive from both, so a structure change (optional
        # field None <-> Field, or the phys carry appearing/disappearing) must
        # rebuild the shard_map rather than reuse stale specs (codex finding,
        # ocean-twin parity). ``phys_state`` threading is the AIMIP-branch
        # feature main lacks (main rejects a non-None carry here).
        key = (jax.tree.structure(c_state),
               None if phys_state is None
               else jax.tree_util.tree_structure(phys_state))
        fn = _cache.get(key)
        if fn is None:
            in_spec = jax.tree.map(lat_spec, c_state)
            if phys_state is None:
                fn = jax.jit(shard_map(
                    _body, mesh=mesh, in_specs=(in_spec, stacks_spec, P()),
                    out_specs=in_spec, check_vma=False))
            else:
                ps_spec = jax.tree.map(_ps_spec_leaf, phys_state)
                fn = jax.jit(shard_map(
                    _body_with_carry, mesh=mesh,
                    in_specs=(in_spec, stacks_spec, P(), ps_spec),
                    out_specs=(in_spec, ps_spec), check_vma=False))
            _cache[key] = fn
        # Arm the SPMD band halo around the call ONLY (see _latlon_spmd_armed).
        with _latlon_spmd_armed(mesh):
            if phys_state is None:
                return fn(c_state, stacks, jnp.asarray(dt))
            return fn(c_state, stacks, jnp.asarray(dt), phys_state)

    sharded_step._geom_stacks = stacks   # test/introspection only
    return sharded_step


def make_sharded_atm_latlon_segment(model, mesh, n_steps: int,
                                    physics_fn=None, *,
                                    shard_geometry: bool = True):
    """Return ``segment(c_state, dt) -> (c_state, all_finite)`` advancing
    ``n_steps`` C-grid steps in ONE compiled program — a ``lax.scan`` of the
    band step inside a single jitted ``shard_map``, built once and reused
    (the M2b lever: "compile atmosphere lat-lon segments instead of
    launching one step at a time").

    Contrast with driving :func:`make_sharded_atm_latlon_step` in a Python
    loop: ONE host dispatch (+ halo-backend arm/restore + cache-key hash) per
    SEGMENT instead of per STEP, and no per-step host round-trip between
    device launches.  The scanned band body is the SAME
    ``_make_band_step_body`` the per-step path runs, so the trajectory
    matches the sequential sharded steps to compilation-order roundoff
    (gated at 1e-12 by ``tests/parallel/test_atm_latlon_segment.py``).

    ``all_finite`` is a REPLICATED traced scalar bool from
    :func:`state_finite_scalar` — the in-graph blowup guard (``psum`` of
    per-band non-finite presence over ALL state leaves).  The host reads
    this ONE scalar per segment instead of gathering the full state.

    ``shard_geometry=True`` (default — a NEW API, no historical layout to
    preserve): each device holds ONLY its own band's geometry slice
    (``P("lat")`` stacks) instead of a replicated all-band copy —
    bit-identical numerics, 1/n_dev the geometry bytes
    (:func:`atm_latlon_geometry_bytes`).

    STATELESS physics only (``None`` / Held-Suarez / column-local closures,
    the production ``run_atm_latlon_spmd`` envelope): a stateful
    ``PhysicsState`` carry is refused loudly — thread it through the
    per-step :func:`make_sharded_atm_latlon_step` until the segment lane
    routes the carry through the scan.

    ``mesh=None``: the single-device twin — ``jit(lax.scan)`` over the serial
    C-grid step with the model's own geometry, same ``(state, all_finite)``
    contract.

    ``n_steps`` is STATIC (the compiled scan length): one compiled program
    per distinct segment length (``run_atm_latlon_spmd`` caches per length —
    at most two: the regular segment and the final remainder).

    Carry dtype: leading steps are UNROLLED outside the ``lax.scan`` until
    the state's dtype signature is a fixed point of the step
    (:func:`unroll_to_dtype_fixed_point` — ``jax.eval_shape`` probe, zero
    FLOPs, trace-time constant).  A mixed-precision IC promotes over the
    first stepS (``p_s`` first, ``u/v/T`` next via the promoted ``p_s`` —
    observed jobs 8916406/8916740) exactly as the per-step Python loop
    absorbs silently; an already-stable state unrolls NOTHING and scans all
    ``n_steps``.  Zero extra casts, zero numerical difference vs the
    per-step lane — never a silent precision change.
    """
    from legoesm.parallel.latlon_spmd import latlon_band_perms
    from legoesm.parallel.shard_map_compat import shard_map
    from legoesm.timestepping.integration import (
        refuse_unthreaded_stateful_physics)

    if int(n_steps) < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    n_steps = int(n_steps)

    def _refuse_carry(phys_state):
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="atm lat-lon compiled segment")
        if phys_state is not None:
            raise NotImplementedError(
                "make_sharded_atm_latlon_segment: a stateful PhysicsState "
                "carry is not yet segment-routed — use the per-step "
                "make_sharded_atm_latlon_step(phys_state=...) path.")

    if mesh is None:                       # single-device compiled segment
        def _serial_seg(c_state, dt):
            def _step1(s):
                out, _ps = model._step_cgrid_impl(
                    s, dt, physics_fn=physics_fn, phys_state=None)
                return out
            # Unroll to the scan-carry dtype fixed point (helper docstring).
            out, n_left = unroll_to_dtype_fixed_point(
                _step1, c_state, n_steps)
            if n_left > 0:
                out, _ = jax.lax.scan(lambda s, _x: (_step1(s), None),
                                      out, xs=None, length=n_left)
            return out, state_finite_scalar(out)

        fn_serial = jax.jit(_serial_seg)

        def serial_segment(c_state, dt, phys_state=None):
            _refuse_carry(phys_state)
            return fn_serial(c_state, jnp.asarray(dt))

        return serial_segment

    n_dev = mesh.devices.size
    axis = mesh.axis_names[0]

    _refuse_unsupported_spmd_config(model)

    template, array_field_names, stacks, stacks_spec = _build_geometry_stacks(
        model, mesh, n_dev, shard_geometry)
    perm_north, _perm_south = latlon_band_perms(n_dev)
    band_step = _make_band_step_body(
        model, template, array_field_names, axis, perm_north, physics_fn,
        shard_geometry)

    def _seg_body(state_local, stacks_local, dt):
        def _step1(s):
            out, _ps = band_step(s, stacks_local, dt, None)
            return out
        # Unroll to the scan-carry dtype fixed point (helper docstring).
        out, n_left = unroll_to_dtype_fixed_point(
            _step1, state_local, n_steps)
        if n_left > 0:
            out, _ = jax.lax.scan(lambda s, _x: (_step1(s), None),
                                  out, xs=None, length=n_left)
        return out, state_finite_scalar(out, axis=axis)

    _cache = {}

    def segment(c_state, dt, phys_state=None):
        _refuse_carry(phys_state)
        # Cache key = state pytree STRUCTURE (in/out specs derive from it) —
        # same doctrine as the per-step factory.
        key = jax.tree.structure(c_state)
        fn = _cache.get(key)
        if fn is None:
            in_spec = jax.tree.map(lat_spec, c_state)
            fn = jax.jit(shard_map(
                _seg_body, mesh=mesh,
                in_specs=(in_spec, stacks_spec, P()),
                out_specs=(in_spec, P()), check_vma=False))
            _cache[key] = fn
        with _latlon_spmd_armed(mesh):
            return fn(c_state, stacks, jnp.asarray(dt))

    segment._geom_stacks = stacks   # test/introspection only
    return segment


def run_atm_latlon_spmd_segment(model, mesh, hs_init, dt, n_steps,
                                physics_fn=None, phys_state=None):
    """Run ``n_steps`` of the lat-band-SPMD C-grid hydrostatic step from a
    cell-centered ``HydrostaticState``, returning a ``HydrostaticState``.

    The Stage-7 driver seam: it bridges the cell-centered driver/output/restart
    contract to the sharded C-grid step. Convert+shard ONCE on entry, run the
    whole ``n_steps`` purely in the sharded C-grid layout, gather+convert ONCE on
    exit. This MATCHES the serial ``for _ in range(n): hs = model.step(hs, dt,
    physics_fn=physics_fn)`` loop, which — via the model's ``id()``-keyed
    ``_cgrid_cache`` — likewise stays in the C-grid layout across the loop and
    only converts cell<->face at the segment boundaries. So the (lossy)
    cell->face->cell round-trip happens ONCE on both paths, not per step, and the
    two trajectories reduce to the Stage-5 C-grid equivalence (tendency bit-exact
    for dynamics + column-local physics; the limited-FV-PPM cut truncation
    bounded).


def get_policy() -> PrecisionPolicy:
    """Return the active global precision policy."""
    return _ACTIVE_POLICY[0]


def validate_policy(policy: PrecisionPolicy | None = None) -> None:
    """Verify the active precision policy is actually achievable.

    On backends that lack float64 (e.g. Metal), float64 requests in the
    policy are silently clamped to float32 by ``resolve_dtype``, so the
    policy is always achievable — this function is a no-op in that case.

    Raises
    ------
    RuntimeError
        If the policy requires float64, the backend supports it, but
        JAX x64 mode is not enabled.
    """
    if policy is None:
        policy = get_policy()
    needs_x64 = jnp.float64 in (
        policy.storage, policy.compute, policy.accumulate, policy.control,
    )
    if not needs_x64:
        return
    if not _backend().supports_float64():
        # Backend cannot do float64; resolve_dtype will clamp to float32.
        return
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "Precision policy requires float64 but JAX x64 mode is not enabled. "
            "Set JAX_ENABLE_X64=1 or call jax.config.update('jax_enable_x64', True) "
            "before bootstrapping."
        )


def set_module_override(module: str, **role_overrides: str) -> None:
    """Override precision roles for a specific module.

    Parameters
    ----------
    module : str
        Module name (e.g., "pressure_gradient", "barotropic_solver").
    **role_overrides
        Role name -> precision mode string. Valid roles: storage, compute,
        accumulate, control. Valid values: "fp32", "fp64", or None to
        clear the override.

    Examples
    --------
    >>> set_module_override("pressure_gradient", compute="fp64", control="fp64")
    >>> set_module_override("tracer_advection", compute="fp32")
    """
    valid_roles = {"storage", "compute", "accumulate", "control"}
    for role in role_overrides:
        if role not in valid_roles:
            raise ValueError(
                f"Unknown role {role!r}. Valid: {sorted(valid_roles)}"
            )

    if module not in _MODULE_OVERRIDES:
        _MODULE_OVERRIDES[module] = {}

    for role, value in role_overrides.items():
        if value is None:
            _MODULE_OVERRIDES[module].pop(role, None)
        else:
            _MODULE_OVERRIDES[module][role] = parse_dtype(
                value, field_name=f"{module}.{role}",
            )


def clear_module_overrides() -> None:
    """Remove all per-module precision overrides."""
    _MODULE_OVERRIDES.clear()


def get_module_overrides() -> dict[str, dict[str, str | None]]:
    """Return current per-module overrides (copy)."""
    return {k: dict(v) for k, v in _MODULE_OVERRIDES.items()}


def _clamp_to_backend(dtype: jnp.dtype) -> jnp.dtype:
    """Clamp *dtype* to what the current backend actually supports.

    On backends that lack float64 (e.g. Apple Metal) or when JAX x64
    mode is disabled, float64 is silently downgraded to float32 so that
    the precision policy never requests an impossible dtype.
    """
    if dtype == jnp.float64:
        b = _backend()
        if not (b.supports_float64() and b.is_x64_enabled()):
            return jnp.float32
    return dtype


def resolve_dtype(module: str | None, role: str) -> jnp.dtype:
    """Resolve the effective dtype for a (module, role) pair.

    Priority: module override > global policy > fallback to float32.
    The result is clamped to what the backend supports (float64 is
    downgraded to float32 on Metal or when x64 is disabled).
    """
    # Check module-level override first.
    if module is not None and module in _MODULE_OVERRIDES:
packages/core/legoesm/core/precision.py:3:Provides a unified precision policy system that controls storage, compute,
packages/core/legoesm/core/precision.py:5:FP32, FP64, and mixed-precision modes for GPU-optimized century-scale
packages/core/legoesm/core/precision.py:8:Three precision modes
packages/core/legoesm/core/precision.py:17:>>> from legoesm.core.precision import PrecisionPolicy, get_policy, cast, const
packages/core/legoesm/core/precision.py:24:- Váňa et al. (2017): Single precision in weather forecasting.
packages/core/legoesm/core/precision.py:37:# runtime.precision, which re-exports from this module — so
packages/core/legoesm/core/precision.py:50:_PRECISION_NAME_TO_DTYPE = {
packages/core/legoesm/core/precision.py:59:    """Parse a precision config value into a JAX dtype."""
packages/core/legoesm/core/precision.py:63:        raise ValueError(f"{field_name} precision cannot be None")
packages/core/legoesm/core/precision.py:68:        if key in _PRECISION_NAME_TO_DTYPE:
packages/core/legoesm/core/precision.py:69:            return _PRECISION_NAME_TO_DTYPE[key]
packages/core/legoesm/core/precision.py:71:        f"Unknown {field_name} precision {value!r}. "
packages/core/legoesm/core/precision.py:72:        f"Use one of {sorted(_PRECISION_NAME_TO_DTYPE.keys())}."
packages/core/legoesm/core/precision.py:81:    """Global precision configuration.
packages/core/legoesm/core/precision.py:123:        """Mixed-precision mode.
packages/core/legoesm/core/precision.py:139:        State arrays are stored in float64 for maximum precision in
packages/core/legoesm/core/precision.py:153:# When a module has a specific precision requirement, it overrides the global
packages/core/legoesm/core/precision.py:165:def set_policy(policy: PrecisionPolicy) -> None:
packages/core/legoesm/core/precision.py:166:    """Set the active global precision policy.
packages/core/legoesm/core/precision.py:182:    """Return the active global precision policy."""
packages/core/legoesm/core/precision.py:187:    """Verify the active precision policy is actually achievable.
packages/core/legoesm/core/precision.py:218:    """Override precision roles for a specific module.
packages/core/legoesm/core/precision.py:225:        Role name -> precision mode string. Valid roles: storage, compute,
packages/core/legoesm/core/precision.py:254:    """Remove all per-module precision overrides."""
packages/core/legoesm/core/precision.py:268:    the precision policy never requests an impossible dtype.
packages/core/legoesm/core/precision.py:303:    This is the primary entry point for precision-aware kernels.
packages/core/legoesm/core/precision.py:308:    precision loss when the default fp32 policy is active but arrays
packages/core/legoesm/core/precision.py:321:        If False (default), skip casts that would reduce precision.
packages/core/legoesm/core/precision.py:365:    This prevents silent precision loss when the default fp32 policy is
packages/core/legoesm/core/precision.py:384:# Reduction wrappers — accumulation-precision aware
packages/core/legoesm/core/precision.py:388:    """Sum with accumulation precision.
packages/core/legoesm/core/precision.py:397:    """Global maximum (precision-independent, returns in input dtype)."""
packages/core/legoesm/core/precision.py:402:    """Vector norm with accumulation precision.
packages/core/legoesm/core/precision.py:433:    """Area-weighted mean with accumulation precision."""
packages/core/legoesm/core/precision.py:478:def with_precision(module: str):
packages/core/legoesm/core/precision.py:483:    @with_precision("tracer_advection")
packages/core/legoesm/core/precision.py:525:# Recommended per-module precision strategies
packages/core/legoesm/core/precision.py:529:# knowledge about which ESM kernels are precision-sensitive.
packages/core/legoesm/core/precision.py:602:    """Apply recommended per-module overrides for a precision mode.
packages/core/legoesm/core/precision.py:605:    global policy.  The caller (``apply_precision``) is responsible for
scripts/bench/bench_atm_latlon_spmd_scaling.py:404:        precision="float64" if jax.config.jax_enable_x64 else "float32",
scripts/bench/bench_atm_latlon_spmd_scaling.py:432:        precision="float64" if jax.config.jax_enable_x64 else "float32",

exec
/usr/bin/bash -lc "sed -n '960,1210p' scripts/bench/metadata.py
sed -n '200,325p' scripts/bench/bench_atm_latlon_spmd_scaling.py
sed -n '1,105p' packages/core/legoesm/core/precision.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 275ms:
      ``rank_imbalance``       median over blocks of the per-block
                               max/median-across-processes ratio (>= 1.0;
                               exactly 1.0 for a single process)
      ``rank_imbalance_per_block``  the per-block ratios themselves
      ``block_steps``/``n_blocks``/``probe_steps``  the EXECUTED schedule
    """
    import time

    import jax
    import numpy as np

    multi = jax.process_count() > 1
    if multi:
        # Collectively verify the schedule BEFORE compilation or any
        # schedule-dependent fence: this allgather is the single collective
        # every process reaches first, so on a mismatch EVERY process sees
        # the same gathered table and raises together instead of hanging in
        # mismatched named fences (codex batch4 deadlock hazard).
        from jax.experimental import multihost_utils
        _sched = np.asarray(multihost_utils.process_allgather(
            np.array([block_steps, n_blocks, probe_steps], dtype=np.int64)))
        if not bool((_sched == _sched[0]).all()):
            raise ValueError(
                "timed_scan_blocks: processes disagree on the schedule "
                f"(block_steps, n_blocks, probe_steps) = {_sched.tolist()} "
                "per process — a mismatched schedule deadlocks in the "
                "named fences.")
    # No silent rewrites: invalid values raise; the recorded schedule IS
    # the executed schedule.  block_steps == 0 stays legal (the documented
    # zero-length parity path).
    if block_steps < 0:
        raise ValueError(
            f"timed_scan_blocks: block_steps must be >= 0, got {block_steps}")
    if n_blocks < 1:
        raise ValueError(
            f"timed_scan_blocks: n_blocks must be >= 1, got {n_blocks}")
    if probe_steps < 0:
        raise ValueError(
            f"timed_scan_blocks: probe_steps must be >= 0, got {probe_steps}")

    def _block(tree):
        jax.block_until_ready(jax.tree_util.tree_leaves(tree))

    def _fence(tag: str):
        if multi:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices(f"{sync_label}_{tag}")

    # --- 1. compile (first call of advance, separated from all timing) ---
    _fence("compile_start")
    t0 = time.perf_counter()
    state = advance(state)
    _block(state)
    compile_ms = (time.perf_counter() - t0) * 1e3

    # --- 2. dispatch-latency probe: individually synced steps, reported
    # separately (NEVER mixed into the fused number) ---
    probe_ms = []
    for _ in range(probe_steps):
        t0 = time.perf_counter()
        state = advance(state)
        _block(state)
        probe_ms.append((time.perf_counter() - t0) * 1e3)
    step_latency_ms = float(np.median(probe_ms)) if probe_ms else float("nan")

    # --- 3. fused scan block (dtype-stable carry, the OM pattern) ---
    input_dtypes = jax.tree_util.tree_map(
        lambda x: x.dtype if hasattr(x, "dtype") else None, state)

    @jax.jit
    def _scan_run(st):
        def _body(carry, _):
            new = advance(carry)
            new = jax.tree_util.tree_map(
                lambda x, d: x.astype(d)
                if d is not None and hasattr(x, "astype") else x,
                new, input_dtypes)
            return new, None
        return jax.lax.scan(_body, st, None, length=block_steps)[0]

    # Pre-compile the scan on a SHALLOW pytree copy: the leaves ALIAS the
    # seed state's arrays (no data copy) — sufficient AND safe because the
    # pre-call only needs matching shapes/dtypes/shardings to warm the
    # compile cache, jitted execution is pure, and ``advance`` is
    # contract-bound not to donate buffers (see the docstring).  The seed
    # value itself is untouched; every process executes the same collective
    # schedule — counts stay matched.
    _pre = jax.tree_util.tree_map(lambda x: x, state)
    t0 = time.perf_counter()
    _pre_out = _scan_run(_pre)
    _block(_pre_out)
    scan_compile_ms = (time.perf_counter() - t0) * 1e3
    del _pre, _pre_out

    block_ms = []
    for b in range(n_blocks):
        _fence(f"block{b}_start")
        t0 = time.perf_counter()
        state = _scan_run(state)
        _block(state)
        block_ms.append((time.perf_counter() - t0) * 1e3)
    _fence("blocks_end")

    # Slowest-rank statistics from the FULL per-block vectors (equal length
    # everywhere — the schedule was collectively verified above).  The
    # parallel time of block b is the slowest process IN that block; a
    # gather of per-process medians would let alternating stragglers make
    # every rank median look fast (codex batch4).
    my_blocks = np.asarray(block_ms, dtype=np.float64)
    if multi:
        from jax.experimental import multihost_utils
        all_blocks = np.asarray(
            multihost_utils.process_allgather(my_blocks))
    else:
        all_blocks = my_blocks[None, :]
    parallel_block_ms = np.max(all_blocks, axis=0)        # (n_blocks,)
    rank_median_ms = np.median(all_blocks, axis=0)        # (n_blocks,)
    with np.errstate(divide="ignore", invalid="ignore"):
        imb_per_block = np.where(rank_median_ms > 0.0,
                                 parallel_block_ms / rank_median_ms,
                                 np.nan)
    rank_imbalance = (float(np.nanmedian(imb_per_block))
                      if np.isfinite(imb_per_block).any() else float("nan"))
    # Headline: median over blocks of the per-block PARALLEL time, per step.
    # A zero-length (parity) block has no per-step time — honest null.
    fused_step_ms = (
        round(float(np.median(parallel_block_ms)) / block_steps, 4)
        if block_steps >= 1 else None)

    metrics = {
        "compile_ms": round(compile_ms, 1),
        "scan_compile_ms": round(scan_compile_ms, 1),
        "step_latency_ms": round(step_latency_ms, 3),
        "block_ms": [round(b, 2) for b in block_ms],
        "parallel_block_ms": [round(float(b), 2) for b in parallel_block_ms],
        "fused_step_ms": fused_step_ms,
        "rank_imbalance": round(rank_imbalance, 4),
        "rank_imbalance_per_block": [round(float(r), 4)
                                     for r in imb_per_block],
        "block_steps": int(block_steps),
        "n_blocks": int(n_blocks),
        "probe_steps": int(probe_steps),
    }
    return state, metrics
    seg_n = int(args.segment_steps)
    if seg_n < 0:
        raise SystemExit(f"--segment-steps must be >= 0, got {seg_n}")
    physics_fn = None
    if args.physics == "held_suarez":
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_latlon
        physics_fn = held_suarez_forcing_latlon

    nd = args.n_devices
    avail = len(jax.devices())
    if avail < nd:
        raise SystemExit(f"need {nd} devices, have {avail} "
                         f"(set --xla_force_host_platform_device_count)")
    if args.multicontroller and nd != avail:
        # A mesh over a strict subset would leave some processes' devices out
        # of the program (non-addressable participation hazard). Route-B uses
        # ALL global devices: one band per device across every process.
        raise SystemExit(
            f"--multicontroller: --n-devices ({nd}) must equal the GLOBAL "
            f"device count ({avail} across {jax.process_count()} processes).")
    n_lat = args.n_lat if args.mode == "strong" else args.nlat_per_dev * nd
    if n_lat % nd != 0:
        raise SystemExit(f"n_lat {n_lat} not divisible by n_devices {nd}")

    if nd == 1:
        model, c0 = _build(n_lat, args.n_lon, args.nlev)
        mesh = None
        c = c0
    else:
        # #1100: band-local IC construction. The nd>1 lanes never materialise
        # the global (n_lat, n_lon, nlev) state per process — each leaf is
        # created via make_array_from_callback for the rows this process's
        # devices own (no global build, no device_put replication, no
        # assert_equal all-gather). This is what lets full-node-packed CPU
        # rungs (128 procs/node) survive at large n_lat.
        model = _build_model(n_lat, args.n_lon, args.nlev)
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:nd]),
                                 axis_names=("lat",))
        c = build_sharded_held_suarez_state_atm_latlon(
            model.grid, model.sigma_coord, mesh)
    if seg_n > 0:
        seg_fn = make_sharded_atm_latlon_segment(
            model, mesh, seg_n, physics_fn=physics_fn)
    else:
        step = make_sharded_atm_latlon_step(model, mesh,
                                            physics_fn=physics_fn)

    per_block_ms = None
    completed_blocks = None
    finite_ok = None   # default fused lane: no in-graph finite check -> null
    timing = None   # metadata.timed_scan_blocks metrics (default lane only)
    if seg_n > 0:
        # Multi-controller: align every process before the timed loop so
        # block wall times aren't skewed by startup jitter (and once after,
        # so no process exits while peers still hold collectives in flight).
        # (The default lane's fences live inside timed_scan_blocks.)
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices("atm_latlon_spmd_bench_start")
        # Segment mode: each timed BLOCK is one compiled lax.scan of seg_n
        # steps; the host sync per block is the production pattern — read the
        # in-graph finite SCALAR, then block on the state for honest timing.
        per_block_ms = []
        finite_ok = True
        for i in range(args.steps):
            t0 = time.perf_counter()
            c, ok = seg_fn(c, args.dt)
            ok_b = bool(ok)
            _block(c)
            per_block_ms.append((time.perf_counter() - t0) * 1e3)
            if not ok_b:
                # A non-finite state poisons every later block: stop timing
                # and mark the whole record invalid — a warning alone let a
                # diverging trajectory serialize as valid scaling data, and
                # an early false was even forgotten by later true blocks
                # (codex batch4).
                finite_ok = False
                print(f"[warn] segment finite scalar FALSE after block {i} "
                      f"(step {(i + 1) * seg_n}) — stopping the timed loop; "
                      "the record is marked INVALID (finite_ok=false, "
                      "valid=false) and its throughput fields are nulled")
                break
        completed_blocks = len(per_block_ms)
        per_step_ms = [b / seg_n for b in per_block_ms]
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices("atm_latlon_spmd_bench_end")
        steady = per_step_ms[args.warmup:]
        if not steady:
            # Divergence stopped the run inside the warmup window — fall back
            # to every completed unit (the record is already marked invalid;
            # this only keeps the diagnostic median well-defined).
            steady = per_step_ms
        med = float(np.median(steady))
    else:
        # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
        # blocks with sync only AROUND the block — the previous per-step
        # host-synced loop measured dispatch+sync latency, not fused device
        # throughput.  Dispatch latency stays measured SEPARATELY
        # (``step_latency_ms``); multi-controller runs record the
        # slowest-process block time + imbalance ratio.
        from metadata import timed_scan_blocks
        c, timing = timed_scan_blocks(
            lambda st: step(st, args.dt), c,
            block_steps=args.steps, n_blocks=args.blocks,
            probe_steps=args.probe_steps,
            sync_label="atm_latlon_spmd_bench")
        # Headline = fused per-step time from the SLOWEST process; key name
        # kept for the aggregators.
        med = float(timing["fused_step_ms"])

    # valid=false ONLY on an observed non-finite state; the default fused
    # lane (finite_ok=None: unchecked) stays valid.
    valid = finite_ok is not False
    # A diverging segment run's med is not a measurement: feed the bound
    # honest nulls (its flat throughput twins are nulled after assembly).
    _measured_med = med if valid else None
    # Honest per-device geometry residency (from the real band-grid shapes):
    # the default lane replicates all-band stacks; the segment lane shards.
    geom_bytes = (atm_latlon_geometry_bytes(model.grid, nd) if nd > 1
                  else None)

    # Communication accounting (audit item 4) + calibrated T_bound (item 8).
    # nd=1: zero inter-device traffic is a FACT (recorded as 0), so the
    # bound is complete and trivially equals the measured compute.  nd>1:
    # there is no analytic halo-message census for the atm latlon step yet
"""Precision management for legoESM.

Provides a unified precision policy system that controls storage, compute,
accumulation, and control dtypes across all model components. Supports
FP32, FP64, and mixed-precision modes for GPU-optimized century-scale
climate simulations.

Three precision modes
---------------------
1. **FP32**: All computation in float32 (fastest, GPU-optimal)
2. **FP64**: Full scientific reference mode (most accurate)
3. **Mixed**: Storage/compute in float32, accumulation/solvers in float64
   (best balance of performance and stability)

Usage
-----
>>> from legoesm.core.precision import PrecisionPolicy, get_policy, cast, const
>>> policy = PrecisionPolicy.mixed()
>>> x_compute = cast(x, "tracer_advection", "compute")
>>> one = const(1.0, "pressure_gradient", "control")

References
----------
- Váňa et al. (2017): Single precision in weather forecasting.
- Klöwer et al. (2020): Number formats, error mitigation, and scope for
  16-bit arithmetics in weather and climate modeling.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Deferred import to break cycle: runtime/__init__.py imports
# runtime.precision, which re-exports from this module — so
# importing runtime.backend at module load time would mid-init
# this very file. Both helpers are only called from function
# bodies, so a lazy import is safe.
def _backend():
    from legoesm.runtime import backend as _b
    return _b


# ---------------------------------------------------------------------------
# Dtype parsing helper
# ---------------------------------------------------------------------------

_PRECISION_NAME_TO_DTYPE = {
    "float16": jnp.float16, "fp16": jnp.float16, "half": jnp.float16,
    "bfloat16": jnp.bfloat16, "bf16": jnp.bfloat16,
    "float32": jnp.float32, "fp32": jnp.float32, "single": jnp.float32,
    "float64": jnp.float64, "fp64": jnp.float64, "double": jnp.float64,
}


def parse_dtype(value, *, field_name: str, allow_none: bool = False):
    """Parse a precision config value into a JAX dtype."""
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} precision cannot be None")
    if value in (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _PRECISION_NAME_TO_DTYPE:
            return _PRECISION_NAME_TO_DTYPE[key]
    raise ValueError(
        f"Unknown {field_name} precision {value!r}. "
        f"Use one of {sorted(_PRECISION_NAME_TO_DTYPE.keys())}."
    )


# ---------------------------------------------------------------------------
# Precision roles
# ---------------------------------------------------------------------------

class PrecisionPolicy(NamedTuple):
    """Global precision configuration.

    Attributes
    ----------
    storage : jnp.dtype
        Dtype for state arrays in memory. Determines GPU memory footprint.
    compute : jnp.dtype
        Dtype for arithmetic in kernels (matmuls, stencils, tendencies).
    accumulate : jnp.dtype
        Dtype for reductions (global sums, budget integrals, norms).
        Must be >= compute width to avoid catastrophic cancellation.
    control : jnp.dtype
        Dtype for solvers, implicit systems, and diagnostics that
        require tight tolerances (semi-implicit, barotropic, EOS).
    """
    storage: jnp.dtype = jnp.float32
    compute: jnp.dtype = jnp.float32
    accumulate: jnp.dtype = jnp.float32
    control: jnp.dtype = jnp.float32

    @staticmethod
    def fp32() -> PrecisionPolicy:
        """All-float32 mode. Fastest, GPU-optimal."""
        return PrecisionPolicy(
            storage=jnp.float32,

exec
/usr/bin/bash -lc "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python - <<'PY'
import runpy, sys, re
sys.argv = ['_probe_latlon_halo_census.py', '8', '512', '1024', '26']
g = runpy.run_path('scripts/tmp/_probe_latlon_halo_census.py')
jax, step, c = g['jax'], g['step'], g['c']
for i in range(4):
    c = step(c, 60.0)
    jax.block_until_ready(c)
    ds = sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(c) if hasattr(x, 'dtype')})
    print('after_step', i + 1, ds)
txt = jax.jit(lambda s: step(s, 60.0)).lower(c).compile().as_text()
DT = {'f32':4,'f64':8,'bf16':2,'f16':2,'s32':4,'u32':4,'pred':1,'s8':1,'u8':1,'c64':8,'c128':16}
items=[]
for m in re.finditer(r'collective-permute(?:-start)?\\\\(', txt):
    line = txt[txt.rfind('\\\\n',0,m.start())+1:m.start()]
    sm = re.search(r'(\\\\w+)\\\\[([0-9,]*)\\\\]', line)
    if sm and sm.group(1) in DT:
        n=1
        for d in sm.group(2).split(','):
            if d:n*=int(d)
        items.append((sm.group(1),n*DT[sm.group(1)], line.strip()))
print('post_warmup_cp_count', len(items), 'total', sum(x[1] for x in items))
from collections import Counter
print('dtype_bytes', Counter((x[0],x[1]) for x in items))
print('sample_lines')
for x in items[:6]: print(x[2])
PY" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 1 in 305ms:
/usr/bin/bash: cannot create temp file for here-document: Read-only file system

exec
/usr/bin/bash -lc 'JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python -c $'"'import runpy, sys\\nsys.argv=[\"_probe_latlon_halo_census.py\",\"8\",\"512\",\"1024\",\"26\"]\\ng=runpy.run_path(\"scripts/tmp/_probe_latlon_halo_census.py\")\\njax, step, c=g[\"jax\"],g[\"step\"],g[\"c\"]\\nfor i in range(4):\\n c=step(c,60.0)\\n jax.block_until_ready(c)\\n print(\"after\",i+1,sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(c) if hasattr(x,\"dtype\")}))'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 15785ms:
{"cp_ops_with_shape": 41, "cp_result_bytes_per_dev_per_step": 4635408, "nd": 8, "n_lat": 512, "n_lon": 1024, "nlev": 26, "rows_per_dev": 64, "census_backend": "virtual-cpu (forced host platform)", "census": {"collective_permute": 41, "all_reduce": 1, "all_gather": 0, "all_to_all": 0, "reduce_scatter": 0, "total": 42}}
after 1 ['float32', 'float64']
after 2 ['float32', 'float64']
after 3 ['float32', 'float64']
after 4 ['float32', 'float64']

exec
/usr/bin/bash -lc 'JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python -c $'"'import runpy, sys, re, collections\\nsys.argv=[\"_probe_latlon_halo_census.py\",\"8\",\"512\",\"1024\",\"26\"]\\ng=runpy.run_path(\"scripts/tmp/_probe_latlon_halo_census.py\")\\njax, step, c=g[\"jax\"],g[\"step\"],g[\"c\"]\\nfor i in range(4):\\n c=step(c,60.0)\\n jax.block_until_ready(c)\\ntxt=jax.jit(lambda s: step(s,60.0)).lower(c).compile().as_text()\\nDT={\"f32\":4,\"f64\":8,\"bf16\":2,\"f16\":2,\"s32\":4,\"u32\":4,\"pred\":1,\"s8\":1,\"u8\":1,\"c64\":8,\"c128\":16}\\nitems=[]\\nfor m in re.finditer(r\"collective-permute(?:-start)?\\\\(\",txt):\\n line=txt[txt.rfind(\"\\\\n\",0,m.start())+1:m.start()]\\n sm=re.search(r\"(\\\\w+)\\\\[([0-9,]*)\\\\]\",line)\\n if sm and sm.group(1) in DT:\\n  n=1\\n  for q in sm.group(2).split(\",\"):\\n   if q: n*=int(q)\\n  items.append((sm.group(1),n*DT[sm.group(1)]))\\nprint(\"post_warmup_cp\",len(items),sum(x[1] for x in items),collections.Counter(items))'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 17102ms:
{"cp_ops_with_shape": 41, "cp_result_bytes_per_dev_per_step": 4635408, "nd": 8, "n_lat": 512, "n_lon": 1024, "nlev": 26, "rows_per_dev": 64, "census_backend": "virtual-cpu (forced host platform)", "census": {"collective_permute": 41, "all_reduce": 1, "all_gather": 0, "all_to_all": 0, "reduce_scatter": 0, "total": 42}}
post_warmup_cp 41 9270800 Counter({('f64', 212992): 13, ('f64', 213200): 6, ('f64', 427648): 6, ('f64', 221616): 6, ('f64', 221184): 6, ('f32', 4): 4})

exec
/usr/bin/bash -lc "PYTHONPATH=scripts/bench /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python -c 'from metadata import calibrated_bound as b; cases=[(\"64f32\",6.7324,2.837,18541632), (\"128f32\",5.5767,1.659,18541632), (\"128f64\",9.6015,2.973,37083264)]; [print(n,b(measured_fused_step_ms=m,single_device_fused_step_ms=c,halo_messages_per_step=41,halo_bytes_per_step=x,n_reductions_per_step=1,rank_imbalance=1.0,latency_us=26.3,bandwidth_GBs=23.5)) for n,m,c,x in cases]'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 324ms:
64f32 {'t_bound_ms': 2.8633, 'measured_over_bound': 2.3513, 'bound_calibrated': True, 'bound_incomplete_reason': None, 'bound_ingredients': {'compute_ms': 2.837, 'comm_ms': 1.867306, 'reduction_ms': 0.0263, 'imbalance_ms': 0.0, 'launch_host_ms': 0.0, 'latency_us': 26.3, 'bandwidth_GBs': 23.5, 'halo_messages_per_step': 41, 'halo_bytes_per_step': 18541632, 'n_reductions_per_step': 1, 'rank_imbalance': 1.0}}
128f32 {'t_bound_ms': 1.8936, 'measured_over_bound': 2.945, 'bound_calibrated': True, 'bound_incomplete_reason': None, 'bound_ingredients': {'compute_ms': 1.659, 'comm_ms': 1.867306, 'reduction_ms': 0.0263, 'imbalance_ms': 0.0, 'launch_host_ms': 0.0, 'latency_us': 26.3, 'bandwidth_GBs': 23.5, 'halo_messages_per_step': 41, 'halo_bytes_per_step': 18541632, 'n_reductions_per_step': 1, 'rank_imbalance': 1.0}}
128f64 {'t_bound_ms': 2.9993, 'measured_over_bound': 3.2012, 'bound_calibrated': True, 'bound_incomplete_reason': None, 'bound_ingredients': {'compute_ms': 2.973, 'comm_ms': 2.656311, 'reduction_ms': 0.0263, 'imbalance_ms': 0.0, 'launch_host_ms': 0.0, 'latency_us': 26.3, 'bandwidth_GBs': 23.5, 'halo_messages_per_step': 41, 'halo_bytes_per_step': 37083264, 'n_reductions_per_step': 1, 'rank_imbalance': 1.0}}

codex
I reproduced the f32 census at both nd=8 and nd=16 and the documented 1024→2048 near-linearity. I also found that the stated @128 f32 bound does not follow from the listed 41-message, 18.5 MB, 26.3 µs/23.5 GB/s inputs; I’m checking the final topology/precision edges before issuing the verdict.
exec
/usr/bin/bash -lc 'JAX_PLATFORMS=cpu /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python scripts/tmp/_probe_latlon_halo_census.py 128 512 1024 26' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 6830ms:
{"cp_ops_with_shape": 41, "cp_result_bytes_per_dev_per_step": 4635408, "nd": 128, "n_lat": 512, "n_lon": 1024, "nlev": 26, "rows_per_dev": 4, "census_backend": "virtual-cpu (forced host platform)", "census": {"collective_permute": 41, "all_reduce": 1, "all_gather": 0, "all_to_all": 0, "reduce_scatter": 0, "total": 42}}

web search: 
web search: arXiv:2607.16100
web search: 
web search: https://arxiv.org/abs/2607.16100
exec
/usr/bin/bash -lc "rg -n -C 12 \"26534060|26630370|6\\.7324|5\\.5767|9\\.6015|LL2048\" docs/performance/scaling/levante_campaign_2026-07-24.md scripts/cluster/scaling_levante scripts/tmp --glob '*.md' --glob '*.sbatch' --glob '*.py'
rg -n -C 5 \"comm_latency|comm_bandwidth|26\\.3|23\\.5|64\\.2|17\\.8\" docs/performance/scaling scripts/cluster/scaling_levante scripts/tmp --glob '*.md' --glob '*.sbatch' --glob '*.py'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 316ms:
docs/performance/scaling/levante_campaign_2026-07-24.md-800-scaling further before plateauing, and it makes the cube's 11.3 ms look
docs/performance/scaling/levante_campaign_2026-07-24.md-801-like a lane-specific overhead rather than something intrinsic to the
docs/performance/scaling/levante_campaign_2026-07-24.md-802-hardware or to SPMD. CAVEAT: these two points come from different jobs
docs/performance/scaling/levante_campaign_2026-07-24.md-803-(26497323, 26498463) — same lane, protocol and day, but not the same-job
docs/performance/scaling/levante_campaign_2026-07-24.md-804-contrast the cube pair enjoyed.
docs/performance/scaling/levante_campaign_2026-07-24.md-805-
docs/performance/scaling/levante_campaign_2026-07-24.md-806-**LAT-LON FIXED-TILE CONTRAST, MEASURED (job 26502539)** — 131.1k
docs/performance/scaling/levante_campaign_2026-07-24.md-807-cols/GPU on both sides:
docs/performance/scaling/levante_campaign_2026-07-24.md-808-
docs/performance/scaling/levante_campaign_2026-07-24.md-809-| arm | devices | cells | ms/step |
docs/performance/scaling/levante_campaign_2026-07-24.md-810-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-811-| LL1024x2048 | 16 | 4.2 M | 5.73 |
docs/performance/scaling/levante_campaign_2026-07-24.md:812:| **LL2048x4096** | **64** | **218.1 M** | **6.73** |
docs/performance/scaling/levante_campaign_2026-07-24.md-813-
docs/performance/scaling/levante_campaign_2026-07-24.md-814-4x the devices carrying 4x the problem costs **+17 %** — weak-scaling
docs/performance/scaling/levante_campaign_2026-07-24.md-815-efficiency **0.85**, sustaining **32.4 GCells/s (506 Mcells/s/GPU) on
docs/performance/scaling/levante_campaign_2026-07-24.md-816-218 million cells**, the campaign's largest atmospheric run by an order
docs/performance/scaling/levante_campaign_2026-07-24.md-817-of magnitude. So communication grows only weakly with device count here
docs/performance/scaling/levante_campaign_2026-07-24.md-818-too (the cube's equivalent contrast was free at 2.25x; lat-lon pays 17 %
docs/performance/scaling/levante_campaign_2026-07-24.md-819-for 4x). Neither lane is comm-limited at these counts — both are limited
docs/performance/scaling/levante_campaign_2026-07-24.md-820-by the per-step fixed cost above.
docs/performance/scaling/levante_campaign_2026-07-24.md-821-
docs/performance/scaling/levante_campaign_2026-07-24.md-822-Scale-out receipts on this lane: LL1536x3072 at 64 GPUs = 4.97 ms (job
docs/performance/scaling/levante_campaign_2026-07-24.md:823:26498266) and the LL2048 point above.
docs/performance/scaling/levante_campaign_2026-07-24.md-824-
docs/performance/scaling/levante_campaign_2026-07-24.md-825-**WHAT THE FIXED TERM IS — ATTRIBUTED (nsys job 26504836): the halo
docs/performance/scaling/levante_campaign_2026-07-24.md-826-exchange, scaling with tile PERIMETER.** Profiling both tiles at the
docs/performance/scaling/levante_campaign_2026-07-24.md-827-SAME 24 GPUs isolates it by subtraction:
docs/performance/scaling/levante_campaign_2026-07-24.md-828-
docs/performance/scaling/levante_campaign_2026-07-24.md-829-| tile | NCCL time / 12 steps | launches | share of GPU time |
docs/performance/scaling/levante_campaign_2026-07-24.md-830-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-831-| C768, 147.5k cols/GPU | 498.7 ms | 1452 | 66.5 % |
docs/performance/scaling/levante_campaign_2026-07-24.md-832-| C512, 65.5k cols/GPU | 280.7 ms | 1128 | 65.4 % |
docs/performance/scaling/levante_campaign_2026-07-24.md-833-
docs/performance/scaling/levante_campaign_2026-07-24.md-834-The comm term grows **1.78x for a 2.25x larger tile AREA** — close to the
docs/performance/scaling/levante_campaign_2026-07-24.md-835-sqrt(2.25) = 1.50x a PERIMETER law predicts, and nowhere near the 2.25x
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1683-   wall-clock brackets logged as overlap evidence. CONFIRM bar:
docs/performance/scaling/levante_campaign_2026-07-24.md-1684-   max(replica) <= 1.10x mean(solo) => guaranteed aggregate >= 3.64x the
docs/performance/scaling/levante_campaign_2026-07-24.md-1685-   32-GPU solo rate (>= 19.9 GC/s if solo reproduces 5.47) = ~3.3x the
docs/performance/scaling/levante_campaign_2026-07-24.md-1686-   observed 128-GPU single-trajectory rate. REFUTE: replica slowdown
docs/performance/scaling/levante_campaign_2026-07-24.md-1687-   >10 % = a CO-EXECUTION penalty, quantified per replica — its
docs/performance/scaling/levante_campaign_2026-07-24.md-1688-   attribution (fabric contention vs placement/topology vs drift) is a
docs/performance/scaling/levante_campaign_2026-07-24.md-1689-   follow-up, not a conclusion of this job.
docs/performance/scaling/levante_campaign_2026-07-24.md-1690-2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
docs/performance/scaling/levante_campaign_2026-07-24.md-1691-   (81.9k/41.0k/20.5k cells/GPU) on the SAME lloyd=0 family, same sfc +
docs/performance/scaling/levante_campaign_2026-07-24.md-1692-   `--reorder-for 128`, same steps/warmup as the s9 ladder. Weak pairs
docs/performance/scaling/levante_campaign_2026-07-24.md-1693-   recomputed only from these.
docs/performance/scaling/levante_campaign_2026-07-24.md-1694-
docs/performance/scaling/levante_campaign_2026-07-24.md:1695:### 3. Recovered phase-2 receipt: lat-lon atmosphere at 128 GPUs (job 26534060, ran 2026-07-30, unanalysed until now)
docs/performance/scaling/levante_campaign_2026-07-24.md-1696-
docs/performance/scaling/levante_campaign_2026-07-24.md:1697:LL2048x4096 L26, same bench + protocol (steps 12 / warmup 3) as the
docs/performance/scaling/levante_campaign_2026-07-24.md-1698-@64 row (job 26502539, f32 6.73 ms):
docs/performance/scaling/levante_campaign_2026-07-24.md-1699-
docs/performance/scaling/levante_campaign_2026-07-24.md-1700-| arm | ms/step | GC/s (col-levels) |
docs/performance/scaling/levante_campaign_2026-07-24.md-1701-|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md:1702:| f32 @128 (65,536 cols/GPU) | 5.5767 | **39.11** |
docs/performance/scaling/levante_campaign_2026-07-24.md:1703:| f64 @128 | 9.6015 | 22.72 |
docs/performance/scaling/levante_campaign_2026-07-24.md-1704-
docs/performance/scaling/levante_campaign_2026-07-24.md-1705-f32 strong 64->128: 1.207x for 2x devices (eff 0.60) with the tile at
docs/performance/scaling/levante_campaign_2026-07-24.md-1706-**65.5k cols/GPU — comfortably ABOVE the ~30k floor** (codex round-21
docs/performance/scaling/levante_campaign_2026-07-24.md-1707-caught the first draft halving this), so the loss is NOT
docs/performance/scaling/levante_campaign_2026-07-24.md-1708-floor-attributable. Mechanism OPEN — candidates (uninstrumented): 1-D
docs/performance/scaling/levante_campaign_2026-07-24.md-1709-band thinning to 16 rows/rank raising halo/compute ratio, and the
docs/performance/scaling/levante_campaign_2026-07-24.md:1710:16 -> 32-node NCCL topology step. 39.11 GC/s (from 5.5767 ms) is the
docs/performance/scaling/levante_campaign_2026-07-24.md-1711-highest measured throughput of ANY lane in the campaign. The companion
docs/performance/scaling/levante_campaign_2026-07-24.md-1712-oc128 (26534067) FAILED pre-#1370-fix with the 109.5 GB resident-args
docs/performance/scaling/levante_campaign_2026-07-24.md-1713-signature; retry submitted post-fix (below).
docs/performance/scaling/levante_campaign_2026-07-24.md-1714-
docs/performance/scaling/levante_campaign_2026-07-24.md-1715-## Hundreds-of-devices push (user directive 2026-08-02)
docs/performance/scaling/levante_campaign_2026-07-24.md-1716-
docs/performance/scaling/levante_campaign_2026-07-24.md-1717-"Push the scaling to hundreds of CPUs and GPUs for lat-lon and MPAS on
docs/performance/scaling/levante_campaign_2026-07-24.md-1718-GPUs." Machine ceiling: 56 nodes x 4 = 224 a100_80 GPUs; compute
docs/performance/scaling/levante_campaign_2026-07-24.md-1719-partition effectively unbounded for our rank counts. Submitted set:
docs/performance/scaling/levante_campaign_2026-07-24.md-1720-
docs/performance/scaling/levante_campaign_2026-07-24.md-1721-| job | what | devices | why |
docs/performance/scaling/levante_campaign_2026-07-24.md-1722-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-1723-| 26628196 | s9 ensemble contention (v3; 26628021/26627810 superseded pre-start) | 128 GPU (4x32) | lever #1 receipt |
docs/performance/scaling/levante_campaign_2026-07-24.md-1724-| 26628071 | oc LL2304 retry post-#1370 | 128 GPU | pre-fix failure was resident-args; predicted PASS at ~0.10 GB/dev residency |
docs/performance/scaling/levante_campaign_2026-07-24.md:1725:| 26628072 | atm LL2304 @96/@192 + LL2880 @192 | 96-192 GPU | LL2048 does not divide 192; LL2880@192 = 86.4k cols/GPU ABOVE floor |
docs/performance/scaling/levante_campaign_2026-07-24.md-1726-| 26628073 | atm lat-lon 2-D pencil r512 np64-512 | 512 CPU ranks | hundreds-of-CPUs lat-lon (wall-pole lane, labelled) |
docs/performance/scaling/levante_campaign_2026-07-24.md-1727-| 26628074 | subdiv-10 lloyd0 prewarm | 1 CPU | unlocks MPAS 128-224 GPUs ABOVE floor (81.9k-46.8k cells/GPU) |
docs/performance/scaling/levante_campaign_2026-07-24.md-1728-| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
docs/performance/scaling/levante_campaign_2026-07-24.md-1729-
docs/performance/scaling/levante_campaign_2026-07-24.md-1730-s10 ladder (128/192/224 GPUs) submits once 26628074's cache lands.
docs/performance/scaling/levante_campaign_2026-07-24.md-1731-
docs/performance/scaling/levante_campaign_2026-07-24.md-1732-### First hundreds receipt in: lat-lon CPU 2-D pencil to 512 ranks (job 26628073)
docs/performance/scaling/levante_campaign_2026-07-24.md-1733-
docs/performance/scaling/levante_campaign_2026-07-24.md-1734-r512 (512x1024 = 524k cols) L26 f64 moist, 32 rpn block:cyclic,
docs/performance/scaling/levante_campaign_2026-07-24.md-1735-wall-pole 2-D pencil lane (labelled; NOT the pole fold):
docs/performance/scaling/levante_campaign_2026-07-24.md-1736-
docs/performance/scaling/levante_campaign_2026-07-24.md-1737-| ranks | cols/rank | ms/step | speedup vs np64 | eff |
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1798-lat-lon lane, using only repo instruments:
docs/performance/scaling/levante_campaign_2026-07-24.md-1799-
docs/performance/scaling/levante_campaign_2026-07-24.md-1800-* **Halo census** (new probe `scripts/tmp/_probe_latlon_halo_census.py`,
docs/performance/scaling/levante_campaign_2026-07-24.md-1801-  virtual-CPU forced-host-platform lowering of the REAL
docs/performance/scaling/levante_campaign_2026-07-24.md-1802-  `make_sharded_atm_latlon_step`): **41 collective-permutes + 1
docs/performance/scaling/levante_campaign_2026-07-24.md-1803-  all-reduce per step**, nd-INDEPENDENT (identical at nd=8 and nd=16 —
docs/performance/scaling/levante_campaign_2026-07-24.md-1804-  the 1-D band structure check). Exact CP payload from compiled-HLO
docs/performance/scaling/levante_campaign_2026-07-24.md-1805-  result shapes: 4,635,408 B/dev/step at n_lon=1024 L26 f32 = 1.06x the
docs/performance/scaling/levante_campaign_2026-07-24.md-1806-  single-row slab model; linear in n_lon (checked 1024 vs 2048, 0.07 %
docs/performance/scaling/levante_campaign_2026-07-24.md-1807-  residual) -> **18.5 MB/dev/step at n_lon=4096 f32**. CAVEAT: CPU
docs/performance/scaling/levante_campaign_2026-07-24.md-1808-  lowering; GPU-side collective combining could change the executed
docs/performance/scaling/levante_campaign_2026-07-24.md-1809-  count (metadata.py:214) — the bound is a MODEL.
docs/performance/scaling/levante_campaign_2026-07-24.md:1810:* **Same-tile nd=1 compute baselines** (job 26630370, roofline recipe):
docs/performance/scaling/levante_campaign_2026-07-24.md-1811-  16x4096 f32 1.659 ms, 32x4096 f32 2.837, 16x4096 f64 2.973.
docs/performance/scaling/levante_campaign_2026-07-24.md-1812-  Approximation, recorded: nd=1 includes pole tiles -> compute term
docs/performance/scaling/levante_campaign_2026-07-24.md-1813-  biased HIGH -> bound conservative.
docs/performance/scaling/levante_campaign_2026-07-24.md-1814-* **Calibrated bound** (`metadata.calibrated_bound`, measured fabric
docs/performance/scaling/levante_campaign_2026-07-24.md-1815-  constants: IB 26.3 us / 23.5 GB/s, NVLink 17.8 / 64.2):
docs/performance/scaling/levante_campaign_2026-07-24.md-1816-
docs/performance/scaling/levante_campaign_2026-07-24.md-1817-| row | measured | t_bound (IB) | measured/bound |
docs/performance/scaling/levante_campaign_2026-07-24.md-1818-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md:1819:| LL2048@64 f32 | 6.732 | 2.863 | **2.35** |
docs/performance/scaling/levante_campaign_2026-07-24.md:1820:| LL2048@128 f32 | 5.577 | 1.848 | **3.02** |
docs/performance/scaling/levante_campaign_2026-07-24.md:1821:| LL2048@128 f64 | 9.602 | 2.999 | **3.20** |
docs/performance/scaling/levante_campaign_2026-07-24.md-1822-
docs/performance/scaling/levante_campaign_2026-07-24.md-1823-* **The lat-lon GPU panel sits ~2.4-3.2x ABOVE its modeled limit** — the
docs/performance/scaling/levante_campaign_2026-07-24.md-1824-  eff-0.60 strong leg is NOT close to the fabric+compute floor. Leading
docs/performance/scaling/levante_campaign_2026-07-24.md-1825-  PLAUSIBLE mechanism (uninstrumented): effective per-CP cost
docs/performance/scaling/levante_campaign_2026-07-24.md-1826-  (launch + schedule + stream sync) well above the raw 26 us fabric
docs/performance/scaling/levante_campaign_2026-07-24.md-1827-  latency across 41 dependency-chained exchanges — the arXiv:2607.16100
docs/performance/scaling/levante_campaign_2026-07-24.md-1828-  small-collective regime. The model itself notes the serialized-latency
docs/performance/scaling/levante_campaign_2026-07-24.md-1829-  vs overlap biases pull opposite ways; treat measured/bound as a
docs/performance/scaling/levante_campaign_2026-07-24.md-1830-  consistency diagnostic, not proven headroom.
docs/performance/scaling/levante_campaign_2026-07-24.md-1831-* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
docs/performance/scaling/levante_campaign_2026-07-24.md:1832:  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
docs/performance/scaling/levante_campaign_2026-07-24.md-1833-  same-job control, falsifiability block in the script. The ocean-lane
docs/performance/scaling/levante_campaign_2026-07-24.md-1834-  null for these flags was reduction-dominated — first atm test.
--
scripts/tmp/fig3_atm128.sbatch-1-#!/bin/bash -l
scripts/tmp/fig3_atm128.sbatch-2-#SBATCH --job-name=atm128
scripts/tmp/fig3_atm128.sbatch-3-#SBATCH --partition=gpu
scripts/tmp/fig3_atm128.sbatch-4-#SBATCH --constraint=a100_80
scripts/tmp/fig3_atm128.sbatch-5-#SBATCH --nodes=32
scripts/tmp/fig3_atm128.sbatch-6-#SBATCH --gpus-per-node=4
scripts/tmp/fig3_atm128.sbatch-7-#SBATCH --exclusive
scripts/tmp/fig3_atm128.sbatch-8-#SBATCH --mem=0
scripts/tmp/fig3_atm128.sbatch-9-#SBATCH --time=06:00:00
scripts/tmp/fig3_atm128.sbatch-10-#SBATCH --output=atm128.%j.log
scripts/tmp/fig3_atm128.sbatch-11-# Figure v3: atmosphere lat-lon at 128 GPUs (the machine's practical max
scripts/tmp/fig3_atm128.sbatch:12:# for one job: 32 of 63 nodes), both precisions, LL2048x4096 L26 =
scripts/tmp/fig3_atm128.sbatch-13-# 65.5k cols/GPU (above the ~30k floor -> should still scale).
scripts/tmp/fig3_atm128.sbatch-14-set -uo pipefail
scripts/tmp/fig3_atm128.sbatch-15-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
scripts/tmp/fig3_atm128.sbatch-16-export JAX_PLATFORMS=cuda,cpu
scripts/tmp/fig3_atm128.sbatch-17-source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/tmp/fig3_atm128.sbatch-18-cd "$REPO"
scripts/tmp/fig3_atm128.sbatch-19-OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/atm128_j${SLURM_JOB_ID}}"
scripts/tmp/fig3_atm128.sbatch-20-mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
scripts/tmp/fig3_atm128.sbatch-21-rc=0
scripts/tmp/fig3_atm128.sbatch-22-for PREC in 0 1; do
scripts/tmp/fig3_atm128.sbatch-23-  P=f32; [ "$PREC" = 1 ] && P=f64
scripts/tmp/fig3_atm128.sbatch:24:  echo "=== atm LL2048x4096 @128 $P ==="
scripts/tmp/fig3_atm128.sbatch-25-  JAX_ENABLE_X64=$PREC srun --ntasks=128 --ntasks-per-node=4 \
scripts/tmp/fig3_atm128.sbatch-26-      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/tmp/fig3_atm128.sbatch-27-    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
scripts/tmp/fig3_atm128.sbatch-28-      --multicontroller --n-devices 128 --mode strong \
scripts/tmp/fig3_atm128.sbatch-29-      --n-lat 2048 --n-lon 4096 --nlev 26 \
scripts/tmp/fig3_atm128.sbatch-30-      --steps 12 --warmup 3 \
scripts/tmp/fig3_atm128.sbatch:31:      --out "$OUTDIR/LL2048_${P}_np128.jsonl" || { echo "$P FAILED"; rc=1; }
scripts/tmp/fig3_atm128.sbatch-32-done
scripts/tmp/fig3_atm128.sbatch-33-for F in "$OUTDIR"/*.jsonl; do
scripts/tmp/fig3_atm128.sbatch-34-  "$PY" -c "
scripts/tmp/fig3_atm128.sbatch-35-import json,os; d=json.loads(open('$F').readline())
scripts/tmp/fig3_atm128.sbatch-36-print(f\"{os.path.basename('$F'):24s} {d['steady_median_ms']:8.2f} ms {d.get('mcells_per_s',0)/1000:.1f} GC/s\")" 2>/dev/null
scripts/tmp/fig3_atm128.sbatch-37-done
scripts/tmp/fig3_atm128.sbatch-38-echo "DONE rc=$rc"; exit $rc
--
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-1-#!/bin/bash -l
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-2-#SBATCH --job-name=ll128_comb
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-3-#SBATCH --account=bb1596_gpu
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-4-#SBATCH --partition=gpu
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-5-#SBATCH --constraint=a100_80
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-6-#SBATCH --nodes=32
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-7-#SBATCH --gpus-per-node=4
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-8-#SBATCH --exclusive
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-9-#SBATCH --mem=0
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-10-#SBATCH --time=01:30:00
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-11-#SBATCH --output=ll128_comb.%j.log
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:12:# CP-COMBINING A/B at LL2048@128 (the measured/bound ~2.9 gap).
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-13-# The calibrated bound (census 41 CP + 1 AR/step, exact bytes 18.5 MB/dev,
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-14-# IB 26.3us/23.5GB/s, nd=1 same-tile compute 1.659 ms) models 1.85-1.89 ms;
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:15:# measured is 5.58 (job 26534060). Leading PLAUSIBLE mechanism: effective
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-16-# per-CP overhead (launch+schedule+sync) >> raw fabric latency across 41
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-17-# dependency-chained exchanges — the arXiv:2607.16100 regime. Lever:
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-18-# COMBINE independent CPs into fewer, larger messages.
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-19-# NOTE: the closed-levers null for these flags was the OCEAN lane
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-20-# (reduction-dominated); this is the first atm-latlon test — not a rerun
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-21-# of a closed null.
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:22:# Falsifiability, BEFORE submit — arms byte-matched to 26534060 protocol:
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-23-#   A control (default flags)      : expect ~5.6 ms
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-24-#   B +cp-combine 8MB threshold    : CONFIRM lever if >=10% under A
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-25-#   C +combine +pipelined-p2p      : scheduling interaction
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-26-#   REFUTE if B,C within 2% of A -> overhead is not combinable-CP count;
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-27-#   next hypothesis = unoverlapped serial chain (scheduling lever).
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-28-set -uo pipefail
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-29-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-30-export JAX_PLATFORMS=cuda,cpu
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-31-export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-32-export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-33-source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-34-cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-35-OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll128_comb_j${SLURM_JOB_ID}}"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-36-mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-37-rc=0
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-38-run_arm () { # tag extra_xla_flags
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:39:  echo "=== LL2048@128 f32 arm=$1 XLA_EXTRA='$2' ==="
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-40-  XLA_FLAGS="${2}" JAX_ENABLE_X64=0 srun --ntasks=128 --ntasks-per-node=4 \
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-41-      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-42-    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-43-      --multicontroller --n-devices 128 --mode strong \
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-44-      --n-lat 2048 --n-lon 4096 --nlev 26 --steps 12 --warmup 3 \
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-45-      --out "$OUTDIR/$1.jsonl" || { echo "$1 FAILED"; rc=1; }
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-46-}
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-47-run_arm A_default ""
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-48-run_arm B_combine "--xla_gpu_collective_permute_combine_threshold_bytes=8388608"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-49-run_arm C_combine_pipelined "--xla_gpu_collective_permute_combine_threshold_bytes=8388608 --xla_gpu_enable_pipelined_p2p=true"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-50-echo "=== RESULTS ==="
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-51-for T in A_default B_combine C_combine_pipelined; do
--
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-1-#!/bin/bash -l
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-2-#SBATCH --job-name=ll_bound
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-3-#SBATCH --account=bb1596_gpu
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-4-#SBATCH --partition=gpu
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-5-#SBATCH --constraint=a100_80
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-6-#SBATCH --nodes=1
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-7-#SBATCH --gpus-per-node=4
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-8-#SBATCH --mem=120G
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-9-#SBATCH --time=01:00:00
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-10-#SBATCH --output=ll_bound.%j.log
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-11-# Single-device SAME-PER-DEVICE-SIZE baselines to complete the calibrated
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch:12:# T_bound for the LL2048 rows (roofline recipe: the nd=1 time must be at
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-13-# the PER-DEVICE tile, not the global grid — passing the global one gives
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-14-# measured/bound < 1, the known tell).  Tiles: 16x4096 (the @128 band),
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-15-# 32x4096 (the @64 band).  APPROXIMATION, recorded: an nd=1 run includes
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-16-# the pole tiles, so its operator mix slightly OVERCOUNTS an interior
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-17-# band's compute -> the bound is conservative (biased high).
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-18-set -uo pipefail
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-19-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-20-export JAX_PLATFORMS=cuda,cpu
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-21-export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-22-export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-23-source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch-24-cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
--
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-1-#!/bin/bash -l
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-2-#SBATCH --job-name=atm_ll_192
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-3-#SBATCH --account=bb1596_gpu
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-4-#SBATCH --partition=gpu
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-5-#SBATCH --constraint=a100_80
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-6-#SBATCH --nodes=48
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-7-#SBATCH --gpus-per-node=4
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-8-#SBATCH --exclusive
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-9-#SBATCH --mem=0
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-10-#SBATCH --time=02:00:00
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-11-#SBATCH --output=atm_ll_192.%j.log
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-12-# HUNDREDS-OF-GPUS lat-lon atmosphere (user directive 2026-08-02).
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:13:# LL2048 does not divide 192 ranks (2048/192 non-integer bands), so the
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-14-# >128 strong pair moves to LL2304 (96 and 192 both divide 2304) and the
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-15-# above-floor "hundreds" point is LL2880x5760 @192 = 86.4k cols/GPU.
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-16-#   arm 1: LL2304x4608 @ 96  f32 (110.6k cols/GPU)
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-17-#   arm 2: LL2304x4608 @192  f32 (55.3k cols/GPU — still ABOVE the ~30k
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-18-#          floor; codex r21 caught the first draft halving these — so
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-19-#          this pair tests device-count cost at healthy tiles)
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-20-#   arm 3: LL2880x5760 @192  f32 (86.4k cols/GPU — the honest 192-GPU
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-21-#          working point; predicted to hold ~existing GC/s levels)
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:22:# Context anchors (same bench, steps 12/warmup 3): LL2048@128 f32
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:23:# 5.58 ms 39.11 GC/s (job 26534060).
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-24-set -uo pipefail
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-25-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-26-export JAX_PLATFORMS=cuda,cpu
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-27-export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-28-export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-29-source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-30-cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-31-OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/atm_ll192_j${SLURM_JOB_ID}}"
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-32-mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-33-rc=0
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-34-run_arm () { # nlat nlon np tag
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-35-  echo "=== atm LL$1x$2 @$3 f32 ==="
scripts/tmp/ocean_crossover_f32_nd4.sbatch-35-    esac
scripts/tmp/ocean_crossover_f32_nd4.sbatch-36-    echo "--- $ARM nd=$ND (f32) ---"
scripts/tmp/ocean_crossover_f32_nd4.sbatch-37-    # shellcheck disable=SC2086
scripts/tmp/ocean_crossover_f32_nd4.sbatch-38-    JAX_ENABLE_X64=0 "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py \
scripts/tmp/ocean_crossover_f32_nd4.sbatch-39-      --n-devices "$ND" --mode strong --n-lat 576 --n-lon 1152 --nlev 20 \
scripts/tmp/ocean_crossover_f32_nd4.sbatch:40:      $EXTRA --comm-latency-us 17.82 --comm-bandwidth-gbs 64.22 \
scripts/tmp/ocean_crossover_f32_nd4.sbatch-41-      --steps 33 --warmup 3 --out "$OUTDIR/${ARM}.jsonl" \
scripts/tmp/ocean_crossover_f32_nd4.sbatch-42-      || { echo "$ARM nd=$ND FAILED"; rc=1; }
scripts/tmp/ocean_crossover_f32_nd4.sbatch-43-  done
scripts/tmp/ocean_crossover_f32_nd4.sbatch-44-done
scripts/tmp/ocean_crossover_f32_nd4.sbatch-45-echo "=== DONE rc=$rc ==="
--
scripts/tmp/ocean_pcg_iters_sweep.sbatch-34-  for ND in 1 4; do
scripts/tmp/ocean_pcg_iters_sweep.sbatch-35-    echo "--- iters=$ITERS nd=$ND ---"
scripts/tmp/ocean_pcg_iters_sweep.sbatch-36-    JAX_ENABLE_X64=1 "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py \
scripts/tmp/ocean_pcg_iters_sweep.sbatch-37-      --n-devices "$ND" --mode strong --n-lat 576 --n-lon 1152 --nlev 20 \
scripts/tmp/ocean_pcg_iters_sweep.sbatch-38-      --baro-solver implicit_cn --force-pcg --pcg-fixed-iters "$ITERS" \
scripts/tmp/ocean_pcg_iters_sweep.sbatch:39:      --comm-latency-us 17.82 --comm-bandwidth-gbs 64.22 \
scripts/tmp/ocean_pcg_iters_sweep.sbatch-40-      --steps 33 --warmup 3 --out "$OUTDIR/iters${ITERS}.jsonl" \
scripts/tmp/ocean_pcg_iters_sweep.sbatch-41-      || { echo "iters=$ITERS nd=$ND FAILED"; rc=1; }
scripts/tmp/ocean_pcg_iters_sweep.sbatch-42-  done
scripts/tmp/ocean_pcg_iters_sweep.sbatch-43-done
scripts/tmp/ocean_pcg_iters_sweep.sbatch-44-echo "=== DONE rc=$rc ==="
--
scripts/tmp/ocean_calibrated_bound.sbatch-7-#SBATCH --gpus-per-node=4
scripts/tmp/ocean_calibrated_bound.sbatch-8-#SBATCH --exclusive
scripts/tmp/ocean_calibrated_bound.sbatch-9-#SBATCH --time=00:40:00
scripts/tmp/ocean_calibrated_bound.sbatch-10-#SBATCH --output=ocean_bound.%j.log
scripts/tmp/ocean_calibrated_bound.sbatch-11-# Re-run the LL576 ladder with the MEASURED NVLink constants from
scripts/tmp/ocean_calibrated_bound.sbatch:12:# bench_ppermute_microbench (job 26457495: latency 17.82 us, bandwidth
scripts/tmp/ocean_calibrated_bound.sbatch:13:# 64.22 GB/s, host dispatch subtracted) so t_bound is calibrated and the
scripts/tmp/ocean_calibrated_bound.sbatch-14-# plots can carry a machine-specific roofline instead of a generic
scripts/tmp/ocean_calibrated_bound.sbatch-15-# linear-ideal line.
scripts/tmp/ocean_calibrated_bound.sbatch-16-set -uo pipefail
scripts/tmp/ocean_calibrated_bound.sbatch-17-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
scripts/tmp/ocean_calibrated_bound.sbatch-18-SCRIPT_DIR="${SUBMIT_DIR}/scripts/cluster/scaling_levante"
--
scripts/tmp/ocean_calibrated_bound.sbatch-29-    [ "$ARM" = wide ] && extra="--baro-solver explicit_substep --wide-halo"
scripts/tmp/ocean_calibrated_bound.sbatch-30-    echo "--- $ARM nd=$ND (calibrated bound) ---"
scripts/tmp/ocean_calibrated_bound.sbatch-31-    # shellcheck disable=SC2086
scripts/tmp/ocean_calibrated_bound.sbatch-32-    JAX_ENABLE_X64=1 "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py \
scripts/tmp/ocean_calibrated_bound.sbatch-33-      --n-devices "$ND" --mode strong --n-lat 576 --n-lon 1152 --nlev 20 \
scripts/tmp/ocean_calibrated_bound.sbatch:34:      $extra --comm-latency-us 17.82 --comm-bandwidth-gbs 64.22 \
scripts/tmp/ocean_calibrated_bound.sbatch-35-      --steps 33 --warmup 3 \
scripts/tmp/ocean_calibrated_bound.sbatch-36-      --out "$OUTDIR/${ARM}.jsonl" || { echo "$ARM nd=$ND FAILED"; rc=1; }
scripts/tmp/ocean_calibrated_bound.sbatch-37-  done
scripts/tmp/ocean_calibrated_bound.sbatch-38-done
scripts/tmp/ocean_calibrated_bound.sbatch-39-echo "=== DONE rc=$rc ==="
--
scripts/tmp/ocean_overlap_test.sbatch-10-#SBATCH --output=ocean_ovl.%j.log
scripts/tmp/ocean_overlap_test.sbatch-11-# Does the residual come from messages the scheduler CANNOT overlap?
scripts/tmp/ocean_overlap_test.sbatch-12-#
scripts/tmp/ocean_overlap_test.sbatch-13-# Established so far: the 4.3-5.5 ms roofline residual is not byte volume
scripts/tmp/ocean_overlap_test.sbatch-14-# (7.2x more bytes = +0.22 ms) and not sharded-formulation work (0 ms on a
scripts/tmp/ocean_overlap_test.sbatch:15:# 1-device mesh). It divides out to 73-83 us/message against 17.8 us
scripts/tmp/ocean_overlap_test.sbatch-16-# isolated. If those messages sit on a DEPENDENCY CHAIN, XLA's
scripts/tmp/ocean_overlap_test.sbatch-17-# latency-hiding scheduler has nothing to hide them behind — so turning it
scripts/tmp/ocean_overlap_test.sbatch-18-# OFF should cost ~nothing. If instead it is currently hiding real work,
scripts/tmp/ocean_overlap_test.sbatch-19-# turning it off should HURT.
scripts/tmp/ocean_overlap_test.sbatch-20-#
--
scripts/tmp/ocean_overlap_test.sbatch-41-        combine)  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_collective_permute_combine_threshold_bytes=33554432" ;;
scripts/tmp/ocean_overlap_test.sbatch-42-      esac
scripts/tmp/ocean_overlap_test.sbatch-43-      JAX_ENABLE_X64=1 "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py \
scripts/tmp/ocean_overlap_test.sbatch-44-        --n-devices "$ND" --mode strong --n-lat 576 --n-lon 1152 --nlev 20 \
scripts/tmp/ocean_overlap_test.sbatch-45-        --baro-solver implicit_cn --force-pcg \
scripts/tmp/ocean_overlap_test.sbatch:46:        --comm-latency-us 17.82 --comm-bandwidth-gbs 64.22 \
scripts/tmp/ocean_overlap_test.sbatch-47-        --steps 33 --warmup 3 --out "$OUTDIR/${ARM}.jsonl"
scripts/tmp/ocean_overlap_test.sbatch-48-    ) || { echo "arm=$ARM nd=$ND FAILED"; rc=1; }
scripts/tmp/ocean_overlap_test.sbatch-49-  done
scripts/tmp/ocean_overlap_test.sbatch-50-done
scripts/tmp/ocean_overlap_test.sbatch-51-echo "=== DONE rc=$rc ==="
--
scripts/tmp/ocean_singlereduce_slope.sbatch-37-      echo "--- variant=$VAR nd=$ND iters=$ITERS ---"
scripts/tmp/ocean_singlereduce_slope.sbatch-38-      JAX_ENABLE_X64=1 "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py \
scripts/tmp/ocean_singlereduce_slope.sbatch-39-        --n-devices "$ND" --mode strong --n-lat 576 --n-lon 1152 --nlev 20 \
scripts/tmp/ocean_singlereduce_slope.sbatch-40-        --baro-solver implicit_cn --force-pcg \
scripts/tmp/ocean_singlereduce_slope.sbatch-41-        --pcg-variant "$VAR" --pcg-fixed-iters "$ITERS" \
scripts/tmp/ocean_singlereduce_slope.sbatch:42:        --comm-latency-us 17.82 --comm-bandwidth-gbs 64.22 \
scripts/tmp/ocean_singlereduce_slope.sbatch-43-        --steps 33 --warmup 3 --out "$OUTDIR/${VAR}_nd${ND}.jsonl" \
scripts/tmp/ocean_singlereduce_slope.sbatch-44-        || { echo "$VAR nd=$ND iters=$ITERS FAILED"; rc=1; }
scripts/tmp/ocean_singlereduce_slope.sbatch-45-    done
scripts/tmp/ocean_singlereduce_slope.sbatch-46-  done
scripts/tmp/ocean_singlereduce_slope.sbatch-47-done
--
scripts/tmp/ocean_roofline_correct.sbatch-25-export JAX_PLATFORMS=cuda
scripts/tmp/ocean_roofline_correct.sbatch-26-source "${SCRIPT_DIR}/_env.sh"
scripts/tmp/ocean_roofline_correct.sbatch-27-cd "$REPO"
scripts/tmp/ocean_roofline_correct.sbatch-28-OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ocean_rl2_$(date +%H%M%S)_j${SLURM_JOB_ID}}"
scripts/tmp/ocean_roofline_correct.sbatch-29-mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
scripts/tmp/ocean_roofline_correct.sbatch:30:LAT=17.82; BW=64.22        # measured NVLink constants, job 26457495
scripts/tmp/ocean_roofline_correct.sbatch-31-rc=0
scripts/tmp/ocean_roofline_correct.sbatch-32-
scripts/tmp/ocean_roofline_correct.sbatch-33-phase1 () {  # rows -> prints fused_step_ms
scripts/tmp/ocean_roofline_correct.sbatch-34-  local rows=$1
scripts/tmp/ocean_roofline_correct.sbatch-35-  JAX_ENABLE_X64=1 "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py \
--
scripts/tmp/ocean_config_headtohead.sbatch-36-    esac
scripts/tmp/ocean_config_headtohead.sbatch-37-    echo "--- $ARM nd=$ND ---"
scripts/tmp/ocean_config_headtohead.sbatch-38-    # shellcheck disable=SC2086
scripts/tmp/ocean_config_headtohead.sbatch-39-    JAX_ENABLE_X64=1 "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py \
scripts/tmp/ocean_config_headtohead.sbatch-40-      --n-devices "$ND" --mode strong --n-lat 576 --n-lon 1152 --nlev 20 \
scripts/tmp/ocean_config_headtohead.sbatch:41:      $EXTRA --comm-latency-us 17.82 --comm-bandwidth-gbs 64.22 \
scripts/tmp/ocean_config_headtohead.sbatch-42-      --check-conservation --cons-rtol 1e-5 \
scripts/tmp/ocean_config_headtohead.sbatch-43-      --steps 33 --warmup 3 --out "$OUTDIR/${ARM}.jsonl" \
scripts/tmp/ocean_config_headtohead.sbatch-44-      || { echo "$ARM nd=$ND FAILED"; rc=1; }
scripts/tmp/ocean_config_headtohead.sbatch-45-  done
scripts/tmp/ocean_config_headtohead.sbatch-46-done
--
scripts/tmp/ocean_full_roofline.sbatch-7-#SBATCH --gpus-per-node=4
scripts/tmp/ocean_full_roofline.sbatch-8-#SBATCH --exclusive
scripts/tmp/ocean_full_roofline.sbatch-9-#SBATCH --time=00:40:00
scripts/tmp/ocean_full_roofline.sbatch-10-#SBATCH --output=ocean_roofline.%j.log
scripts/tmp/ocean_full_roofline.sbatch-11-# COMPLETE calibrated roofline: measured NVLink constants (job 26457495:
scripts/tmp/ocean_full_roofline.sbatch:12:# 17.82 us / 64.22 GB/s, dispatch-subtracted) AND the single-device compute
scripts/tmp/ocean_full_roofline.sbatch-13-# term (job 26457919: icn 63.559 ms, wide 71.541 ms at LL576 nd=1). With
scripts/tmp/ocean_full_roofline.sbatch-14-# all three ingredients present t_bound is finally emitted non-null and
scripts/tmp/ocean_full_roofline.sbatch-15-# measured_over_bound becomes meaningful — the first machine-specific
scripts/tmp/ocean_full_roofline.sbatch-16-# theoretical limit in this campaign.
scripts/tmp/ocean_full_roofline.sbatch-17-set -uo pipefail
--
scripts/tmp/ocean_full_roofline.sbatch-28-  local arm=$1 nd=$2 sdev=$3; shift 3
scripts/tmp/ocean_full_roofline.sbatch-29-  echo "--- $arm nd=$nd (single-dev $sdev ms) ---"
scripts/tmp/ocean_full_roofline.sbatch-30-  # shellcheck disable=SC2086
scripts/tmp/ocean_full_roofline.sbatch-31-  JAX_ENABLE_X64=1 "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py \
scripts/tmp/ocean_full_roofline.sbatch-32-    --n-devices "$nd" --mode strong --n-lat 576 --n-lon 1152 --nlev 20 \
scripts/tmp/ocean_full_roofline.sbatch:33:    "$@" --comm-latency-us 17.82 --comm-bandwidth-gbs 64.22 \
scripts/tmp/ocean_full_roofline.sbatch-34-    --single-dev-fused-ms "$sdev" \
scripts/tmp/ocean_full_roofline.sbatch-35-    --steps 33 --warmup 3 \
scripts/tmp/ocean_full_roofline.sbatch-36-    --out "$OUTDIR/${arm}.jsonl" || { echo "$arm nd=$nd FAILED"; rc=1; }
scripts/tmp/ocean_full_roofline.sbatch-37-}
scripts/tmp/ocean_full_roofline.sbatch-38-for ND in 2 4; do
--
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-9-#SBATCH --mem=0
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-10-#SBATCH --time=01:30:00
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-11-#SBATCH --output=ll128_comb.%j.log
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-12-# CP-COMBINING A/B at LL2048@128 (the measured/bound ~2.9 gap).
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-13-# The calibrated bound (census 41 CP + 1 AR/step, exact bytes 18.5 MB/dev,
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:14:# IB 26.3us/23.5GB/s, nd=1 same-tile compute 1.659 ms) models 1.85-1.89 ms;
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-15-# measured is 5.58 (job 26534060). Leading PLAUSIBLE mechanism: effective
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-16-# per-CP overhead (launch+schedule+sync) >> raw fabric latency across 41
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-17-# dependency-chained exchanges — the arXiv:2607.16100 regime. Lever:
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-18-# COMBINE independent CPs into fewer, larger messages.
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-19-# NOTE: the closed-levers null for these flags was the OCEAN lane
--
docs/performance/scaling/scaling_crm_gpu.md-22-
docs/performance/scaling/scaling_crm_gpu.md-23-**Profile (N=96, fp64) via `scripts/bench/profile_crm_step.py`:**
docs/performance/scaling/scaling_crm_gpu.md-24-
docs/performance/scaling/scaling_crm_gpu.md-25-| stage                       | ms/step | % full step |
docs/performance/scaling/scaling_crm_gpu.md-26-|-----------------------------|---------|-------------|
docs/performance/scaling/scaling_crm_gpu.md:27:| full_step                   | 17.86   | 100         |
docs/performance/scaling/scaling_crm_gpu.md-28-| acoustic_substeps × 12 (1 RK3 stage) | 6.96    | 39          |
docs/performance/scaling/scaling_crm_gpu.md-29-| acoustic_substeps × 12 × 3  | 20.88   | ~94         |
docs/performance/scaling/scaling_crm_gpu.md-30-| slow_tend (1 RK3 stage)     | 0.31    | 1.8         |
docs/performance/scaling/scaling_crm_gpu.md-31-| slow_tend × 3 (full RK3)    | 0.94    | 5           |
docs/performance/scaling/scaling_crm_gpu.md-32-| hyperdiff_biharmonic        | 0.019   | 0.1         |
--
docs/performance/scaling/scaling_crm_gpu.md-234-
docs/performance/scaling/scaling_crm_gpu.md-235-Extended `scripts/bench/bench_crm_gpu_scaling.py` to print inline HBM
docs/performance/scaling/scaling_crm_gpu.md-236-bandwidth utilization estimate alongside each run:
docs/performance/scaling/scaling_crm_gpu.md-237-
docs/performance/scaling/scaling_crm_gpu.md-238-```
docs/performance/scaling/scaling_crm_gpu.md:239:N 128 step=1.18ms throughput=417.8Mc/s SYPD=4.65 HBM≈702GB/s(96%)
docs/performance/scaling/scaling_crm_gpu.md-240-N 192 step=2.76ms throughput=400.4Mc/s SYPD=1.98 HBM≈673GB/s(92%)
docs/performance/scaling/scaling_crm_gpu.md-241-N 256 step=6.55ms throughput=300.1Mc/s SYPD=0.84 HBM≈504GB/s(69%)
docs/performance/scaling/scaling_crm_gpu.md-242-```
docs/performance/scaling/scaling_crm_gpu.md-243-
docs/performance/scaling/scaling_crm_gpu.md-244-Formula: `Mc/s × 80 B/cell-lev × passes_per_step` where
--
docs/performance/scaling/scaling_crm_gpu.md-606-
docs/performance/scaling/scaling_crm_gpu.md-607-Bench fp32 nsub=6:
docs/performance/scaling/scaling_crm_gpu.md-608-| N   | iter-66 (vmap rm) | iter-67 (hoist) | Δ      |
docs/performance/scaling/scaling_crm_gpu.md-609-|-----|-------------------|-----------------|--------|
docs/performance/scaling/scaling_crm_gpu.md-610-| 128 | 276.8             | **284.8**       | +2.9%  |
docs/performance/scaling/scaling_crm_gpu.md:611:| 192 | 264.2             | **278.6**       | +5.5%  |
docs/performance/scaling/scaling_crm_gpu.md-612-| 256 | 254.9             | 252.9           | noise  |
docs/performance/scaling/scaling_crm_gpu.md-613-| 384 | 155.8             | 156.4           | noise  |
docs/performance/scaling/scaling_crm_gpu.md-614-
docs/performance/scaling/scaling_crm_gpu.md-615-Bit-for-bit on random tridiag fp64 (both implicit_buoyancy=False
docs/performance/scaling/scaling_crm_gpu.md-616-and =True): max_diff a/b/c = **0.0**.
--
docs/performance/scaling/scaling_crm_gpu.md-648-
docs/performance/scaling/scaling_crm_gpu.md-649-Verified:
docs/performance/scaling/scaling_crm_gpu.md-650-- Bit-for-bit identical to legacy `_thomas_solve_batched_legacy`
docs/performance/scaling/scaling_crm_gpu.md-651-  on random tridiag fp64: `max_diff=4.4e-16`, `max_residual=1.3e-15`
docs/performance/scaling/scaling_crm_gpu.md-652-- Bench fp32 nsub=6: N=128 276.8 → 276.8 Mc/s,
docs/performance/scaling/scaling_crm_gpu.md:653:  N=192 264.2 → 264.2 Mc/s, N=256 254.9 → 254.9 Mc/s
docs/performance/scaling/scaling_crm_gpu.md-654-- XLA was already optimizing the vmap, so this is perf-neutral
docs/performance/scaling/scaling_crm_gpu.md-655-  but cleaner (saves the surrounding reshape boilerplate)
docs/performance/scaling/scaling_crm_gpu.md-656-
docs/performance/scaling/scaling_crm_gpu.md-657-Standalone cuSPARSE timing (n_cols=16384, nlev=30, fp32):
docs/performance/scaling/scaling_crm_gpu.md-658-- jitted `tridiagonal_solve(..., d[...,None])[...,0]` = **37 us/call**
--
docs/performance/scaling/levante_campaign_2026-07-24.md-140-serialized "loop fusion" kernels appear — loop_add_fusion_1/2 at 3.6 ms
docs/performance/scaling/levante_campaign_2026-07-24.md-141-per launch (vs ~3 us for ordinary elementwise kernels) plus a THIRD
docs/performance/scaling/levante_campaign_2026-07-24.md-142-once-per-step group (the unsuffixed loop_add_fusion: 12 of its 44
docs/performance/scaling/levante_campaign_2026-07-24.md-143-instances are >1 ms at ~3.3 ms, the rest are the ordinary us-scale adds)
docs/performance/scaling/levante_campaign_2026-07-24.md-144-— and the sqlite timeline places all three groups' big instances at the
docs/performance/scaling/levante_campaign_2026-07-24.md:145:17.8 ms step cadence (stddev 78 us: deterministic compute, not comm
docs/performance/scaling/levante_campaign_2026-07-24.md-146-wait): 3.6 + 3.6 + 3.3 ~= 10.5-10.9 ms/step = the np4 excess; (3) in the optimized
docs/performance/scaling/levante_campaign_2026-07-24.md-147-step HLO these are mega-fusions ON THE HALO PATH: `%loop_add_fusion =
docs/performance/scaling/levante_campaign_2026-07-24.md-148-f32[491520,26]` (edge-tendency add chain, 22 operands incl. an
docs/performance/scaling/levante_campaign_2026-07-24.md-149-input_scatter_fusion) and `%loop_add_fusion.4 = f32[163844,26]` (cell
docs/performance/scaling/levante_campaign_2026-07-24.md-150-array), with the shard_map halo-pack concatenates taking the same adds +
--
docs/performance/scaling/levante_campaign_2026-07-24.md-233-and cited the f64 small-base numbers; the direction and size of the effect
docs/performance/scaling/levante_campaign_2026-07-24.md-234-are unchanged, but the comparison is only valid precision-matched. So the earlier weak ladder measured a below-floor tile rather than a code
docs/performance/scaling/levante_campaign_2026-07-24.md-235-limit, and the same
docs/performance/scaling/levante_campaign_2026-07-24.md-236-config that fixes strong scaling also carries weak (+0.15 at nd4). Ideal is
docs/performance/scaling/levante_campaign_2026-07-24.md-237-flat; the improved arm holds 22.5→22.9 ms while production drifts
docs/performance/scaling/levante_campaign_2026-07-24.md:238:17.8→25.3 ms.
docs/performance/scaling/levante_campaign_2026-07-24.md-239-
docs/performance/scaling/levante_campaign_2026-07-24.md-240-## "It used to be faster / did we regress?" — resolved, no regression
docs/performance/scaling/levante_campaign_2026-07-24.md-241-
docs/performance/scaling/levante_campaign_2026-07-24.md-242-- Cross-machine anchor (matched bench/config/grid/physics/precision,
docs/performance/scaling/levante_campaign_2026-07-24.md-243-  job 26450081): Levante single A100-80 latlon-moist r720 f32 =
--
docs/performance/scaling/levante_campaign_2026-07-24.md-337-claim is made from it.
docs/performance/scaling/levante_campaign_2026-07-24.md-338-
docs/performance/scaling/levante_campaign_2026-07-24.md-339-## DISTANCE TO THE THEORETICAL LIMIT, measured (job 26457977)
docs/performance/scaling/levante_campaign_2026-07-24.md-340-
docs/performance/scaling/levante_campaign_2026-07-24.md-341-With all three ingredients measured on this machine — fabric constants
docs/performance/scaling/levante_campaign_2026-07-24.md:342:(17.82 us, 64.22 GB/s) and the per-tile single-device compute term — the
docs/performance/scaling/levante_campaign_2026-07-24.md-343-ocean LL576 f64 implicit ladder finally has a real roofline:
docs/performance/scaling/levante_campaign_2026-07-24.md-344-
docs/performance/scaling/levante_campaign_2026-07-24.md-345-| nd | measured | calibrated bound | measured/bound | at % of floor |
docs/performance/scaling/levante_campaign_2026-07-24.md-346-|---|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-347-| 2 | 42.71 ms | 34.77 ms | 1.228 | 81 % |
docs/performance/scaling/levante_campaign_2026-07-24.md:348:| 4 | 23.53 ms | 16.82 ms | 1.398 | 72 % |
docs/performance/scaling/levante_campaign_2026-07-24.md-349-
docs/performance/scaling/levante_campaign_2026-07-24.md-350-Bound = per-device compute (32.58 / 14.63 ms) + modelled comm (2.21) +
docs/performance/scaling/levante_campaign_2026-07-24.md-351-modelled reduction (2.19). The unmodelled gap is **~5 ms/step and roughly
docs/performance/scaling/levante_campaign_2026-07-24.md-352-FLAT** with device count (7.9 ms at nd2, 6.7 at nd4), which is why the
docs/performance/scaling/levante_campaign_2026-07-24.md-353-ratio worsens as compute shrinks.
--
docs/performance/scaling/levante_campaign_2026-07-24.md-357-devices). The bench's `comm_scope_note` correctly warns that its census is
docs/performance/scaling/levante_campaign_2026-07-24.md-358-"barotropic implicit-CN PCG scope only … baroclinic 3-D pads NOT counted",
docs/performance/scaling/levante_campaign_2026-07-24.md-359-and the true volume IS much larger: **16.22 MB/step across 110
docs/performance/scaling/levante_campaign_2026-07-24.md-360-collective-permutes vs the censused 2.25 MB — a 7.2x undercount**. But
docs/performance/scaling/levante_campaign_2026-07-24.md-361-completing the census moves the bound by only **0.22 ms**, because the
docs/performance/scaling/levante_campaign_2026-07-24.md:362:comm term is LATENCY-dominated: at 122 messages x 17.82 us the latency part
docs/performance/scaling/levante_campaign_2026-07-24.md:363:is 2.174 ms while even 16 MB at 64.22 GB/s is just 0.253 ms.
docs/performance/scaling/levante_campaign_2026-07-24.md-364-
docs/performance/scaling/levante_campaign_2026-07-24.md-365-So with the byte census completed the unexplained residual is still 5.5 ms
docs/performance/scaling/levante_campaign_2026-07-24.md-366-(nd2) and 4.3 ms (nd4).
docs/performance/scaling/levante_campaign_2026-07-24.md-367-
docs/performance/scaling/levante_campaign_2026-07-24.md-368-SECOND CANDIDATE ALSO REFUTED (`scripts/tmp/probe_sharded_overhead.py`,
--
docs/performance/scaling/levante_campaign_2026-07-24.md-379-Zero within noise at both tiles, so the bound's compute term is the RIGHT
docs/performance/scaling/levante_campaign_2026-07-24.md-380-reference and extra sharded work is not the gap.
docs/performance/scaling/levante_campaign_2026-07-24.md-381-
docs/performance/scaling/levante_campaign_2026-07-24.md-382-WHERE THAT LEAVES IT (quantified, one candidate standing): the residual
docs/performance/scaling/levante_campaign_2026-07-24.md-383-divided by the message count is **83 us/message at nd2 and 73 us at nd4**,
docs/performance/scaling/levante_campaign_2026-07-24.md:384:versus **17.8 us** for the same collective measured in isolation — an in-
docs/performance/scaling/levante_campaign_2026-07-24.md-385-context cost 4-5x the best case. That is consistent with EXPOSED,
docs/performance/scaling/levante_campaign_2026-07-24.md-386-un-overlapped communication rather than raw wire time.
docs/performance/scaling/levante_campaign_2026-07-24.md-387-
docs/performance/scaling/levante_campaign_2026-07-24.md-388-THIRD CANDIDATE REFUTED, AND IT IDENTIFIES THE MECHANISM (job 26458930).
docs/performance/scaling/levante_campaign_2026-07-24.md-389-If the residual were communication the scheduler is currently hiding work
docs/performance/scaling/levante_campaign_2026-07-24.md-390-behind, DISABLING XLA's latency-hiding scheduler would hurt. It does not:
docs/performance/scaling/levante_campaign_2026-07-24.md-391-
docs/performance/scaling/levante_campaign_2026-07-24.md-392-| arm | nd2 | nd4 | vs default |
docs/performance/scaling/levante_campaign_2026-07-24.md-393-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md:394:| default (LHS on) | 42.69 | 23.53 ms | — |
docs/performance/scaling/levante_campaign_2026-07-24.md-395-| `latency_hiding_scheduler=false` | 42.44 | 23.49 | **+0.6 % / +0.2 %** |
docs/performance/scaling/levante_campaign_2026-07-24.md:396:| `enable_pipelined_p2p=true` | 42.76 | 23.54 | -0.2 % / -0.1 % |
docs/performance/scaling/levante_campaign_2026-07-24.md:397:| CP combining @32 MiB | 42.55 | 23.59 | +0.3 % / -0.3 % |
docs/performance/scaling/levante_campaign_2026-07-24.md-398-
docs/performance/scaling/levante_campaign_2026-07-24.md-399-Turning overlap OFF is free (marginally faster), and no scheduling flag
docs/performance/scaling/levante_campaign_2026-07-24.md-400-moves the step. The scheduler has nothing to hide the comm behind.
docs/performance/scaling/levante_campaign_2026-07-24.md-401-
docs/performance/scaling/levante_campaign_2026-07-24.md-402-**MECHANISM (the three tested alternatives are not dominant): the
--
docs/performance/scaling/levante_campaign_2026-07-24.md-432-`--pcg-fixed-iters` was wired onto the SPMD bench for this (the MPI twin
docs/performance/scaling/levante_campaign_2026-07-24.md-433-already had it) and swept at LL576 f64:
docs/performance/scaling/levante_campaign_2026-07-24.md-434-
docs/performance/scaling/levante_campaign_2026-07-24.md-435-| iters | nd1 ms | nd4 ms | eff@4 | speedup@4 | residual (nd4) |
docs/performance/scaling/levante_campaign_2026-07-24.md-436-|---|---|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md:437:| 60 (default) | 63.55 | 23.52 | 0.68 | 1.00x | 2.1e-04 |
docs/performance/scaling/levante_campaign_2026-07-24.md-438-| 40 | 61.33 | 21.05 | 0.73 | 1.12x | 1.6e-03 |
docs/performance/scaling/levante_campaign_2026-07-24.md-439-| 30 | 60.24 | 19.89 | 0.76 | 1.18x | 4.6e-03 |
docs/performance/scaling/levante_campaign_2026-07-24.md-440-| 20 | 59.08 | 18.60 | 0.79 | 1.26x | 1.3e-02 |
docs/performance/scaling/levante_campaign_2026-07-24.md-441-| 10 | 57.99 | 17.31 | **0.84** | **1.36x** | 4.1e-02 |
docs/performance/scaling/levante_campaign_2026-07-24.md-442-
--
docs/performance/scaling/levante_campaign_2026-07-24.md-478-| term | ms | note |
docs/performance/scaling/levante_campaign_2026-07-24.md-479-|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-480-| same-tile single-device step (144 rows) | 14.65 | measured |
docs/performance/scaling/levante_campaign_2026-07-24.md-481-| + PCG dependent sync (60 x 95.7 us) | **5.74** | **65 % of the distributed overhead** |
docs/performance/scaling/levante_campaign_2026-07-24.md-482-| + iteration-independent remainder | 3.13 | 35 % — NOT attributed; may include fixed PCG/setup work |
docs/performance/scaling/levante_campaign_2026-07-24.md:483:| = total | 23.52 | measured 23.52 |
docs/performance/scaling/levante_campaign_2026-07-24.md-484-
docs/performance/scaling/levante_campaign_2026-07-24.md-485-So of the 8.87 ms the step pays for being distributed across 4 GPUs, TWO
docs/performance/scaling/levante_campaign_2026-07-24.md-486-THIRDS is dependent PCG synchronisation and one third is everything else.
docs/performance/scaling/levante_campaign_2026-07-24.md-487-Codex objection (c) is honoured in the same table: the non-PCG part is
docs/performance/scaling/levante_campaign_2026-07-24.md-488-real, separated, and not attributed to the solver.
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1002-  on this dycore — it alters RK2/RK3 boundary tendencies. Exact
docs/performance/scaling/levante_campaign_2026-07-24.md-1003-  communication avoidance would need a 3 x radius = 6-cell overlap with
docs/performance/scaling/levante_campaign_2026-07-24.md-1004-  cube-edge interpolation, and the tiled transport supports only halo
docs/performance/scaling/levante_campaign_2026-07-24.md-1005-  1/2. That is a new algorithm, not a port.
docs/performance/scaling/levante_campaign_2026-07-24.md-1006-
docs/performance/scaling/levante_campaign_2026-07-24.md:1007:**THE BOUND (one step, 86 SendRecv, latency floor 17.8 us measured):**
docs/performance/scaling/levante_campaign_2026-07-24.md-1008-
docs/performance/scaling/levante_campaign_2026-07-24.md-1009-| class | n | total |
docs/performance/scaling/levante_campaign_2026-07-24.md-1010-|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-1011-| <=25 us (latency-bound) | 39 | **0.61 ms** |
docs/performance/scaling/levante_campaign_2026-07-24.md-1012-| 25-100 us | 33 | 1.20 ms |
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1015-Field packing removes LAUNCHES, so its absolute ceiling is the
docs/performance/scaling/levante_campaign_2026-07-24.md-1016-latency-bound class: **0.61 ms of a 17.11 ms step = 3.6 %**. Not worth
docs/performance/scaling/levante_campaign_2026-07-24.md-1017-the change.
docs/performance/scaling/levante_campaign_2026-07-24.md-1018-
docs/performance/scaling/levante_campaign_2026-07-24.md-1019-**What the tail actually is.** The largest single exchange is **4.86 ms**.
docs/performance/scaling/levante_campaign_2026-07-24.md:1020:At the measured 64.2 GB/s NVLink that would be ~310 MB, but the entire
docs/performance/scaling/levante_campaign_2026-07-24.md-1021-per-step halo volume is ~8 MB. So that call is not moving data — it is
docs/performance/scaling/levante_campaign_2026-07-24.md-1022-WAITING. Fourteen exchanges holding 7.54 ms is arrival skew absorbed at
docs/performance/scaling/levante_campaign_2026-07-24.md-1023-halo sync points, the same signature codex flagged for the single-shot
docs/performance/scaling/levante_campaign_2026-07-24.md-1024-all-reduce.
docs/performance/scaling/levante_campaign_2026-07-24.md-1025-
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1394-`bound_calibrated=false`:
docs/performance/scaling/levante_campaign_2026-07-24.md-1395-
docs/performance/scaling/levante_campaign_2026-07-24.md-1396-| lane | devices | latency | bandwidth | vs line rate |
docs/performance/scaling/levante_campaign_2026-07-24.md-1397-|---|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-1398-| NVLink (1 process) | 2 | 18.0 us | 53.98 GB/s | — |
docs/performance/scaling/levante_campaign_2026-07-24.md:1399:| NVLink (1 process) | 4 | 17.8 us | 64.22 GB/s | — |
docs/performance/scaling/levante_campaign_2026-07-24.md:1400:| NCCL over IB (8 procs, 2 nodes) | 8 | 26.3 us | 23.53 GB/s | **94 % of HDR200's 25 GB/s** |
docs/performance/scaling/levante_campaign_2026-07-24.md-1401-
docs/performance/scaling/levante_campaign_2026-07-24.md-1402-The IB number landing at 94 % of line rate is the independent check that the
docs/performance/scaling/levante_campaign_2026-07-24.md-1403-collective itself — not something else — is being timed.
docs/performance/scaling/levante_campaign_2026-07-24.md-1404-
docs/performance/scaling/levante_campaign_2026-07-24.md-1405-HOST DISPATCH MUST BE SUBTRACTED, and this was nearly a self-inflicted
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1495-2. **MPAS ico np4 per-device dip** — five hypotheses refuted by
docs/performance/scaling/levante_campaign_2026-07-24.md-1496-   measurement (see above); needs a GPU op-level profile (nsys / XLA op
docs/performance/scaling/levante_campaign_2026-07-24.md-1497-   profile of np4 vs np8). A scoped instrumentation project, not a knob.
docs/performance/scaling/levante_campaign_2026-07-24.md-1498-3. ~~Calibrated theoretical-limit lines~~ **DONE later in this campaign**:
docs/performance/scaling/levante_campaign_2026-07-24.md-1499-   `bench_ppermute_microbench.py` measured the fabric constants (NVLink
docs/performance/scaling/levante_campaign_2026-07-24.md:1500:   17.8 us / 64.2 GB/s, IB 26.3 us / 23.5 GB/s) and the roofline sections
docs/performance/scaling/levante_campaign_2026-07-24.md-1501-   above use them. Kept here only so the list's numbering stays stable.
docs/performance/scaling/levante_campaign_2026-07-24.md-1502-4. **Ocean wet-cell compaction + wet-balanced partitions** — SPLIT
docs/performance/scaling/levante_campaign_2026-07-24.md-1503-   2026-07-26 into a cheap half and an expensive half:
docs/performance/scaling/levante_campaign_2026-07-24.md-1504-
docs/performance/scaling/levante_campaign_2026-07-24.md-1505-   *Cheap half — wet-BALANCED bands* (no indirection, uneven band heights
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1810-* **Same-tile nd=1 compute baselines** (job 26630370, roofline recipe):
docs/performance/scaling/levante_campaign_2026-07-24.md-1811-  16x4096 f32 1.659 ms, 32x4096 f32 2.837, 16x4096 f64 2.973.
docs/performance/scaling/levante_campaign_2026-07-24.md-1812-  Approximation, recorded: nd=1 includes pole tiles -> compute term
docs/performance/scaling/levante_campaign_2026-07-24.md-1813-  biased HIGH -> bound conservative.
docs/performance/scaling/levante_campaign_2026-07-24.md-1814-* **Calibrated bound** (`metadata.calibrated_bound`, measured fabric
docs/performance/scaling/levante_campaign_2026-07-24.md:1815:  constants: IB 26.3 us / 23.5 GB/s, NVLink 17.8 / 64.2):
docs/performance/scaling/levante_campaign_2026-07-24.md-1816-
docs/performance/scaling/levante_campaign_2026-07-24.md-1817-| row | measured | t_bound (IB) | measured/bound |
docs/performance/scaling/levante_campaign_2026-07-24.md-1818-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-1819-| LL2048@64 f32 | 6.732 | 2.863 | **2.35** |
docs/performance/scaling/levante_campaign_2026-07-24.md-1820-| LL2048@128 f32 | 5.577 | 1.848 | **3.02** |
--
docs/performance/scaling/amip_mpi_scaling.md-68-Strong scaling (fixed I5, vary ranks):
docs/performance/scaling/amip_mpi_scaling.md-69-
docs/performance/scaling/amip_mpi_scaling.md-70-| ranks | HS ms/step | HS speedup | dycore-only ms/step |
docs/performance/scaling/amip_mpi_scaling.md-71-|------:|-----------:|-----------:|--------------------:|
docs/performance/scaling/amip_mpi_scaling.md-72-| 1     | 29.23      | 1.00×      | 28.29 |
docs/performance/scaling/amip_mpi_scaling.md:73:| 2     | 23.58      | 1.24× (62%)| 22.35 |
docs/performance/scaling/amip_mpi_scaling.md-74-| 4     | 24.22      | 1.21× (30%)| 22.41 |
docs/performance/scaling/amip_mpi_scaling.md-75-
docs/performance/scaling/amip_mpi_scaling.md-76-Weak (const ~66k cells/rank, apples-to-apples): rank1 I4 = 6.51 ms vs
docs/performance/scaling/amip_mpi_scaling.md-77-rank4 I5 = 23.70 ms → 3.6× slower → ~27% weak efficiency.
docs/performance/scaling/amip_mpi_scaling.md-78-
--
docs/performance/scaling/amip_mpi_scaling.md-90-
docs/performance/scaling/amip_mpi_scaling.md-91-Profiled the icosahedral dynamics MPI step. Ruled out, in order:
docs/performance/scaling/amip_mpi_scaling.md-92-
docs/performance/scaling/amip_mpi_scaling.md-93-- **Thread oversubscription**: single-thread-per-rank (`XLA_FLAGS=
docs/performance/scaling/amip_mpi_scaling.md-94-  --xla_cpu_multi_thread_eigen=false`, `OMP_NUM_THREADS=1`) gives the *same*
docs/performance/scaling/amip_mpi_scaling.md:95:  plateau (I5 HS 28.97→23.59→24.08→25.73 ms for 1/2/4/8 ranks). NOT the cause.
docs/performance/scaling/amip_mpi_scaling.md-96-  Corollary: the MPAS CPU step barely multithreads (many small gather/scatter
docs/performance/scaling/amip_mpi_scaling.md-97-  ops), so more-threads doesn't help either.
docs/performance/scaling/amip_mpi_scaling.md-98-- **Raw MPI latency**: allreduce 15–78 µs, sendrecv-8KB 6–16 µs (np 2/4/8).
docs/performance/scaling/amip_mpi_scaling.md-99-  ~170 µs total/step — negligible.
docs/performance/scaling/amip_mpi_scaling.md-100-- **mpi4jax-in-JIT and -in-scan cost**: 0.77–2.0 ms/step (python-loop),

exec
/usr/bin/bash -lc "sed -n '1,280p' scripts/cluster/scaling_levante/_env.sh
sed -n '800,850p' docs/performance/scaling/levante_campaign_2026-07-24.md
sed -n '1390,1415p' docs/performance/scaling/levante_campaign_2026-07-24.md
rg -n -C 3 \"XLA_FLAGS\" scripts/cluster/scaling_levante/_env.sh scripts/cluster/scaling_levante/*.sbatch | sed -n '1,360p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 244ms:
# Shared environment for DKRZ Levante GPU scaling jobs (sourced by the SLURM
# scripts here).  SLURM/OpenMPI twin of scripts/cluster/scaling_derecho/_env.sh
# (which is PBS/Cray-MPICH).
# ---------------------------------------------------------------------------
# EDIT the marked values (account / repo / conda env / module versions) before
# the first submit.  Everything is overridable from the sbatch environment, e.g.
#   sbatch --export=ALL,LEGOESM_CONDA_ENV=my-jax-env gpu_moist_scaling.slurm
#
# Levante GPU partition (partition `gpu`): 60 nodes, each 2x AMD EPYC 7763 +
# 4x NVIDIA A100 (56 nodes 80GB, 4 nodes 40GB), InfiniBand HDR200.  MPI stack is
# OpenMPI over UCX with CUDA-aware transports -- NOT Cray MPICH.
# ---------------------------------------------------------------------------

# --- (1) Project allocation (matches SBATCH --account in the job scripts) -----
#     Levante GPU jobs bill a *_gpu sub-account (run_levante_gpu_scaling.sh uses
#     bd1083_gpu, matching the SBATCH --account in the .slurm, which is the
#     source of truth).  This default is only for interactive sourcing.
export LEGOESM_SLURM_ACCOUNT="${LEGOESM_SLURM_ACCOUNT:-bd1083_gpu}"

# --- (2) Repo location on Levante -- EDIT to where you cloned legoESM ---------
REPO="${LEGOESM_REPO:-/work/bd1083/$USER/legoESM}"
export REPO

# --- (3) Conda env with a CUDA jaxlib AND a CUDA-aware mpi4jax (see README) ---
CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"

# --- Federation PYTHONPATH (belt-and-braces; `pip install -e .` makes it
#     redundant but harmless) ------------------------------------------------
PP="$REPO/src"
for p in atmosphere core coupler ice land ml ocean tools; do
  PP="$PP:$REPO/packages/$p"
done
export PYTHONPATH="$PP:${PYTHONPATH:-}"

# --- Modules + conda -- EDIT the module versions to the Levante stack you built
#     mpi4py / mpi4jax against (README Step 1); pinned versions matter because
#     the runtime libmpi ABI must match the build ABI ------------------------
module load python3 2>/dev/null || true      # EDIT: e.g. python3/2023.01-gcc-11.2.0
module load openmpi 2>/dev/null || true       # EDIT: the CUDA-aware openmpi you built against
module load cuda    2>/dev/null || true       # EDIT: matching cuda toolkit
if command -v conda >/dev/null 2>&1; then
  conda activate "$CONDA_ENV" 2>/dev/null || true
fi
PY="${LEGOESM_PYTHON:-$(command -v python)}"
export PY

# --- JAX / runtime knobs -----------------------------------------------------
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export MPI4JAX_NO_WARN_JAX_VERSION=1
export MPLBACKEND="${MPLBACKEND:-Agg}"          # headless plotting
# DKRZ scratch is /scratch/<first-letter-of-user>/<user>.
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
# Persistent JIT cache reuses compiles across runs; set empty to force a cold
# compile (true compile_time_s).  On SCRATCH so it survives between jobs.
export LEGOESM_JIT_CACHE_DIR="${LEGOESM_JIT_CACHE_DIR:-$SCRATCH/legoesm_jit_cache}"

# --- OpenMPI + UCX CUDA-aware fabric (GPU route-A) ---------------------------
# Route-A hands the on-device sendrecv buffer straight to MPI (the whole point:
# no device->host->device staging, which would erase multi-GPU scaling).  On
# Levante that path is OpenMPI-over-UCX; the pml/osc + UCX transports below turn
# on GPU-direct: cuda_copy + cuda_ipc intra-node, gdr_copy over InfiniBand HDR
# inter-node.  Requires a CUDA-aware mpi4jax (README) + MPI4JAX_USE_CUDA_MPI=1
# (set in the job script).  UCX_MEMTYPE_CACHE=n avoids a stale device/host
# memtype-cache hang that CUDA-aware sendrecv is prone to.
export OMPI_MCA_pml="${OMPI_MCA_pml:-ucx}"
export OMPI_MCA_osc="${OMPI_MCA_osc:-ucx}"
export UCX_TLS="${UCX_TLS:-rc,cuda_copy,cuda_ipc,gdr_copy,sm,self}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export UCX_RNDV_SCHEME="${UCX_RNDV_SCHEME:-put_zcopy}"

# --- NCCL over InfiniBand (route-B: jax.distributed multi-node lanes) --------
# NCCL (shard_map/ppermute collectives under jax.distributed) uses its own
# IB-verbs stack — independent of the UCX/MPI settings above; the two configs
# coexist. Bootstrap ring runs over IPoIB: verify the interface name once with
# `ip addr` on a gpu node (a wrong NCCL_SOCKET_IFNAME is the #1 cause of
# multi-node NCCL bootstrap timeouts on IB clusters).
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
# Prefix-match BOTH HCAs (mlx5_0/mlx5_1 — one per socket on Levante nodes).
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
# GPUDirect RDMA when NIC and GPU share a NUMA/PCIe root.
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"

export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
mkdir -p "$TMPDIR" 2>/dev/null || true
scaling further before plateauing, and it makes the cube's 11.3 ms look
like a lane-specific overhead rather than something intrinsic to the
hardware or to SPMD. CAVEAT: these two points come from different jobs
(26497323, 26498463) — same lane, protocol and day, but not the same-job
contrast the cube pair enjoyed.

**LAT-LON FIXED-TILE CONTRAST, MEASURED (job 26502539)** — 131.1k
cols/GPU on both sides:

| arm | devices | cells | ms/step |
|---|---|---|---|
| LL1024x2048 | 16 | 4.2 M | 5.73 |
| **LL2048x4096** | **64** | **218.1 M** | **6.73** |

4x the devices carrying 4x the problem costs **+17 %** — weak-scaling
efficiency **0.85**, sustaining **32.4 GCells/s (506 Mcells/s/GPU) on
218 million cells**, the campaign's largest atmospheric run by an order
of magnitude. So communication grows only weakly with device count here
too (the cube's equivalent contrast was free at 2.25x; lat-lon pays 17 %
for 4x). Neither lane is comm-limited at these counts — both are limited
by the per-step fixed cost above.

Scale-out receipts on this lane: LL1536x3072 at 64 GPUs = 4.97 ms (job
26498266) and the LL2048 point above.

**WHAT THE FIXED TERM IS — ATTRIBUTED (nsys job 26504836): the halo
exchange, scaling with tile PERIMETER.** Profiling both tiles at the
SAME 24 GPUs isolates it by subtraction:

| tile | NCCL time / 12 steps | launches | share of GPU time |
|---|---|---|---|
| C768, 147.5k cols/GPU | 498.7 ms | 1452 | 66.5 % |
| C512, 65.5k cols/GPU | 280.7 ms | 1128 | 65.4 % |

The comm term grows **1.78x for a 2.25x larger tile AREA** — close to the
sqrt(2.25) = 1.50x a PERIMETER law predicts, and nowhere near the 2.25x
an area law would give. That is the mechanism behind the fitted "fixed"
11.27 ms: halo cost tracks the tile EDGE while compute tracks the tile
AREA, so under strong scaling compute falls off faster than communication
and the comm share rises until it dominates. It is NOT launch overhead
and NOT host synchronisation.

This also reconciles with the fixed-tile contrast (comm does not grow
with DEVICE COUNT at constant tile, 1.06x for 2.25x devices): both
statements are true because the comm cost is set by the tile geometry,
not by how many devices exist.

**LANE-MISMATCH CORRECTION (found while reading the kernel names).**
That profile omitted `--closed-loop`, so it characterised the SINGLE-SHOT
adapter lane (49.95 ms at C768), whereas the 11.27 ms fixed-cost model
was fitted to CLOSED-LOOP runs (19.33 ms). I had attributed the 2.6x gap

`scripts/bench/bench_ppermute_microbench.py` (new) times the SAME collective
the sharded steps use — `lax.ppermute` on a ring inside `shard_map` — over a
message-size sweep, so the SPMD benches can stop reporting
`bound_calibrated=false`:

| lane | devices | latency | bandwidth | vs line rate |
|---|---|---|---|---|
| NVLink (1 process) | 2 | 18.0 us | 53.98 GB/s | — |
| NVLink (1 process) | 4 | 17.8 us | 64.22 GB/s | — |
| NCCL over IB (8 procs, 2 nodes) | 8 | 26.3 us | 23.53 GB/s | **94 % of HDR200's 25 GB/s** |

The IB number landing at 94 % of line rate is the independent check that the
collective itself — not something else — is being timed.

HOST DISPATCH MUST BE SUBTRACTED, and this was nearly a self-inflicted
error: the FIRST version timed one jit call per exchange and reported
287-518 us "latency" (job 26457469) — two orders above what these fabrics
do, because dispatch dominates a single call. Feeding that into `t_bound`
would have produced a confidently WRONG roofline, strictly worse than the
uncalibrated generic line it was meant to replace. The tool now times 1 rep
vs n reps inside one jit and differences them; the removed dispatch
(287-529 us) is still reported alongside so the contamination stays
visible. Validated on CPU virtual devices: 275 us -> 26 us.

Quote the lane that matches the plot: intra-node NVLink and inter-node IB
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-239-        (
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-240-            case "$ARM" in
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-241-              fused) export LEGOESM_LATLON_SPMD_FUSED_HALO=1 ;;
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:242:              xla)   export XLA_FLAGS="${XLA_FLAGS:-} ${_XLA_COMM_FLAGS}" ;;
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-243-              pgle)  export JAX_ENABLE_PGLE=true JAX_PGLE_PROFILING_RUNS=3 ;;
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-244-            esac
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-245-            srun --ntasks=8 --ntasks-per-node=4 $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
--
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-272-    for ARM in base xla; do
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-273-        echo "--- lane T arm=$ARM: cube cs-spmd C$CS_RES np=6 ---"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-274-        (
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:275:            [ "$ARM" = xla ] && export XLA_FLAGS="${XLA_FLAGS:-} ${_XLA_COMM_FLAGS}"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-276-            srun --ntasks=6 --ntasks-per-node=3 $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-277-                "$PY" scripts/bench/run_cpu_mpi_scaling.py \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-278-                --grid cubed-sphere --cs-spmd --device gpu \
--
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-37-rc=0
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-38-run_arm () { # tag extra_xla_flags
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-39-  echo "=== LL2048@128 f32 arm=$1 XLA_EXTRA='$2' ==="
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:40:  XLA_FLAGS="${2}" JAX_ENABLE_X64=0 srun --ntasks=128 --ntasks-per-node=4 \
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-41-      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-42-    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-43-      --multicontroller --n-devices 128 --mode strong \
--
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-59-fi
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-60-
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-61-echo "== Stage A: CPU-virtual parity gates =="
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:62:JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=24 \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-63-srun -N1 -n1 "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-64-    --kt "${KT}" --resolution 48 --nlev 10 --steps 4 --warmup 1 \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-65-    --parity-gate --out "${OUT%.jsonl}_parity.jsonl"
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:66:JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=24 \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-67-srun -N1 -n1 "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-68-    --kt "${KT}" --resolution 48 --nlev 10 --steps 4 --warmup 1 \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-69-    --closed-loop --parity-gate --out "${OUT%.jsonl}_loop_parity.jsonl"
--
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-76-# GPU defaults those clique kinds init concurrently and can be ordered
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-77-# differently per rank -> NCCL comm-init deadlock. Force serial comm init (see
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-78-# cube_tiled_step.pbs for the full rationale + bisection).
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:79:export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_latency_hiding_scheduler=false"
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-80-timeout 1800 srun "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-81-    --kt "${KT}" --resolution "${RES}" --nlev "${NLEV}" --dt "${DT}" \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-82-    --steps 12 --warmup 2 --multicontroller --out "${OUT}"

exec
/usr/bin/bash -lc "sed -n '320,425p' docs/performance/scaling/levante_campaign_2026-07-24.md
sed -n '425,510p' docs/performance/scaling/levante_campaign_2026-07-24.md
sed -n '1,180p' scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 304ms:

Shape: ~1.0 through np8, one step down, then FLAT 0.5 from np16 to np64 —
4x more ranks with no further loss. CONSISTENT with a fixed per-rank cost
rather than growing communication, but flat efficiency alone does not
identify which; that needs phase-level timing. (The >1 points at np8 are the same
base-leg-working-set effect noted for the cube; read as "at ideal".)

Against the 4-node SPREAD lat-lon ladder (job 26452578) at high rank
counts the contrast is large: ico holds 0.52 at np64 where lat-lon r128/r256
is at 0.12/0.16. That matches the documented expectation that a 2-D cell
partition beats a 1-D latitude band on perimeter/area. CAVEAT: ico ran
PACKED on one node and lat-lon SPREAD over four, so this compares
decomposition AND placement together, not decomposition alone.

The ico WEAK ladder from the same job is non-monotone (1.00 / 0.47 / 0.81 /
0.48 / 0.46 / 0.22 / 0.45 at np 1..64) — the per-rank problem size is not
held constant cleanly across that sweep's subdivision steps, so no weak
claim is made from it.

## DISTANCE TO THE THEORETICAL LIMIT, measured (job 26457977)

With all three ingredients measured on this machine — fabric constants
(17.82 us, 64.22 GB/s) and the per-tile single-device compute term — the
ocean LL576 f64 implicit ladder finally has a real roofline:

| nd | measured | calibrated bound | measured/bound | at % of floor |
|---|---|---|---|---|
| 2 | 42.71 ms | 34.77 ms | 1.228 | 81 % |
| 4 | 23.53 ms | 16.82 ms | 1.398 | 72 % |

Bound = per-device compute (32.58 / 14.63 ms) + modelled comm (2.21) +
modelled reduction (2.19). The unmodelled gap is **~5 ms/step and roughly
FLAT** with device count (7.9 ms at nd2, 6.7 at nd4), which is why the
ratio worsens as compute shrinks.

WHAT THE GAP IS NOT — the omitted-traffic explanation is REFUTED
(`scripts/tmp/probe_ocean_halo_bytes.py`, HLO byte census on CPU virtual
devices). The bench's `comm_scope_note` correctly warns that its census is
"barotropic implicit-CN PCG scope only … baroclinic 3-D pads NOT counted",
and the true volume IS much larger: **16.22 MB/step across 110
collective-permutes vs the censused 2.25 MB — a 7.2x undercount**. But
completing the census moves the bound by only **0.22 ms**, because the
comm term is LATENCY-dominated: at 122 messages x 17.82 us the latency part
is 2.174 ms while even 16 MB at 64.22 GB/s is just 0.253 ms.

So with the byte census completed the unexplained residual is still 5.5 ms
(nd2) and 4.3 ms (nd4).

SECOND CANDIDATE ALSO REFUTED (`scripts/tmp/probe_sharded_overhead.py`,
job 26458553): the sharded formulation does NOT do measurably more work.
Timing the SHARDED step on a 1-device mesh (all the padding, band-edge and
v-row-reconstruction machinery present, ppermutes self-to-self so no real
traffic) against the UNSHARDED step at the identical tile:

| tile | unsharded | sharded on 1 device | overhead |
|---|---|---|---|
| 288x1152x20 | 33.13 ms | 32.79 ms | **-0.34 ms (-1.0 %)** |
| 144x1152x20 | 15.88 ms | 15.92 ms | **+0.04 ms (+0.2 %)** |

Zero within noise at both tiles, so the bound's compute term is the RIGHT
reference and extra sharded work is not the gap.

WHERE THAT LEAVES IT (quantified, one candidate standing): the residual
divided by the message count is **83 us/message at nd2 and 73 us at nd4**,
versus **17.8 us** for the same collective measured in isolation — an in-
context cost 4-5x the best case. That is consistent with EXPOSED,
un-overlapped communication rather than raw wire time.

THIRD CANDIDATE REFUTED, AND IT IDENTIFIES THE MECHANISM (job 26458930).
If the residual were communication the scheduler is currently hiding work
behind, DISABLING XLA's latency-hiding scheduler would hurt. It does not:

| arm | nd2 | nd4 | vs default |
|---|---|---|---|
| default (LHS on) | 42.69 | 23.53 ms | — |
| `latency_hiding_scheduler=false` | 42.44 | 23.49 | **+0.6 % / +0.2 %** |
| `enable_pipelined_p2p=true` | 42.76 | 23.54 | -0.2 % / -0.1 % |
| CP combining @32 MiB | 42.55 | 23.59 | +0.3 % / -0.3 % |

Turning overlap OFF is free (marginally faster), and no scheduling flag
moves the step. The scheduler has nothing to hide the comm behind.

**MECHANISM (the three tested alternatives are not dominant): the
residual is best explained by EXPOSED, DEPENDENCY-SERIALIZED
SYNCHRONISATION.** These are eliminations of the TESTED implementations,
not of every possible communication explanation. Not bytes
(7.2x more = +0.22 ms), not sharded-formulation work (0 ms), not
hideable-by-scheduling (0 ms). It is the unavoidable cost of sync points
that sit on a dependent chain.

This retro-explains every earlier arm in the campaign, which is the check
that the mechanism is right rather than merely last-standing:
- fused-halo NULL — aggregation reduces message COUNT but not chain DEPTH;
- `single_reduce` HELPED (0.49 -> 0.53) — Chronopoulos-Gear restructures
  the recurrence into fewer DEPENDENT reduction batches;
- wide-halo HELPED MOST (-> 0.73) — it deletes the barotropic solver's sync
  points outright;
- the f64/f32 flip — more compute per sync point dilutes a fixed sync cost.

ACTIONABLE CONSEQUENCE: the lever for this lane is reducing the NUMBER OF
DEPENDENT SYNCHRONISATION POINTS, not message aggregation, byte
compression, or XLA scheduling flags — three families this campaign has
now measured to be null here.

Honest answer to "how far from the theoretical limit are we": 72-81 % of a
now-calibrated floor, with the shortfall attributable to neither bandwidth
now-calibrated floor, with the shortfall attributable to neither bandwidth
nor byte volume.

## The mechanism's prediction, TESTED — and the lever it exposes (job 26459382)

If exposed dependent sync is the cost, PCG iteration count is the most
direct lever on it (each iteration carries dependent reduction batches).
`--pcg-fixed-iters` was wired onto the SPMD bench for this (the MPI twin
already had it) and swept at LL576 f64:

| iters | nd1 ms | nd4 ms | eff@4 | speedup@4 | residual (nd4) |
|---|---|---|---|---|---|
| 60 (default) | 63.55 | 23.52 | 0.68 | 1.00x | 2.1e-04 |
| 40 | 61.33 | 21.05 | 0.73 | 1.12x | 1.6e-03 |
| 30 | 60.24 | 19.89 | 0.76 | 1.18x | 4.6e-03 |
| 20 | 59.08 | 18.60 | 0.79 | 1.26x | 1.3e-02 |
| 10 | 57.99 | 17.31 | **0.84** | **1.36x** | 4.1e-02 |

QUANTIFICATION — and codex round-7 rates the strong form OVERSTATED, which
is recorded here rather than argued away. Regressing T(N) = intercept +
N x slope over the five iteration counts:

| | intercept | slope | R^2 |
|---|---|---|---|
| nd=1 | 56.87 +- 0.04 ms | 111.4 +- 1.0 us/iter | 0.99994 |
| nd=4 | 16.11 +- 0.09 ms | 123.8 +- 2.4 us/iter | 0.99972 |

WHAT THIS ESTABLISHES (codex objection (c), answered): the fit is
essentially exact and the intercept is tightly determined, so the cost is
genuinely PER-ITERATION, not a constant misattributed to iterations. But
the nd4 intercept EXCEEDS the measured 144-row compute term (14.63 ms) by
**1.48 ms**, so a fixed non-PCG overhead does exist and the 60 iterations
do NOT explain the entire residual.

CODEX OBJECTION (a) TESTED AND REFUTED (job 26459817). Rather than divide
the nd1 slope by 4, measure the per-iteration slope directly at each tile
on ONE device:

| tile | measured us/iter |
|---|---|
| 576 rows | 110.0 |
| 288 rows | 65.2 |
| 144 rows | **28.1** |

The linearity assumption predicted 110.0/4 = 27.5 us for the 144-row tile;
the MEASURED value is 28.1 us — 2 % apart. Per-iteration compute IS linear
in rows (the PCG iteration is a bandwidth-bound stencil+reduction, so it
scales with data even where the FULL step does not). The sync figure barely
moves: **95.7 us/iter measured** vs 96.3 assumed.

DECOMPOSITION OF THE nd=4 STEP (both terms from measured slopes; the
remainder is left UNATTRIBUTED):

| term | ms | note |
|---|---|---|
| same-tile single-device step (144 rows) | 14.65 | measured |
| + PCG dependent sync (60 x 95.7 us) | **5.74** | **65 % of the distributed overhead** |
| + iteration-independent remainder | 3.13 | 35 % — NOT attributed; may include fixed PCG/setup work |
| = total | 23.52 | measured 23.52 |

So of the 8.87 ms the step pays for being distributed across 4 GPUs, TWO
THIRDS is dependent PCG synchronisation and one third is everything else.
Codex objection (c) is honoured in the same table: the non-PCG part is
real, separated, and not attributed to the solver.

STRONG-SCALING CONSEQUENCE: cutting iterations raises 4-GPU efficiency from
0.68 to 0.84, because what is being removed is precisely the part that does
NOT shrink with device count.

THE CATCH, WHICH IS NOT MINE TO WAIVE: the speed is bought with solver
convergence — the zero-forcing probe residual degrades 200x from 2.1e-04 to
4.1e-02. 40 iterations (1.12x, 8x residual) and 30 (1.18x, 22x) are the
plausible operating points, but whether that residual is acceptable for the
free surface is an OCEAN-SCIENCE decision, not a performance one. Reported
as a trade curve; no default changed.

## The accuracy-free lever, and what it reveals (job 26460113)

Cutting PCG iterations trades accuracy. `single_reduce`
(Chronopoulos-Gear) attacks the SAME sync at unchanged iteration count by
halving the DEPENDENT reduction batches per iteration, so it should be
free. Measured at LL576 f64, slopes from a 60-vs-20-iteration difference:

| variant | nd | us/iter | step @60 | residual |
|---|---|---|---|---|
| standard | 1 | 111.0 | 63.54 ms | 2.079e-04 |
#!/bin/bash -l
#SBATCH --job-name=ll128_comb
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=ll128_comb.%j.log
# CP-COMBINING A/B at LL2048@128 (the measured/bound ~2.9 gap).
# The calibrated bound (census 41 CP + 1 AR/step, exact bytes 18.5 MB/dev,
# IB 26.3us/23.5GB/s, nd=1 same-tile compute 1.659 ms) models 1.85-1.89 ms;
# measured is 5.58 (job 26534060). Leading PLAUSIBLE mechanism: effective
# per-CP overhead (launch+schedule+sync) >> raw fabric latency across 41
# dependency-chained exchanges — the arXiv:2607.16100 regime. Lever:
# COMBINE independent CPs into fewer, larger messages.
# NOTE: the closed-levers null for these flags was the OCEAN lane
# (reduction-dominated); this is the first atm-latlon test — not a rerun
# of a closed null.
# Falsifiability, BEFORE submit — arms byte-matched to 26534060 protocol:
#   A control (default flags)      : expect ~5.6 ms
#   B +cp-combine 8MB threshold    : CONFIRM lever if >=10% under A
#   C +combine +pipelined-p2p      : scheduling interaction
#   REFUTE if B,C within 2% of A -> overhead is not combinable-CP count;
#   next hypothesis = unoverlapped serial chain (scheduling lever).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll128_comb_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
run_arm () { # tag extra_xla_flags
  echo "=== LL2048@128 f32 arm=$1 XLA_EXTRA='$2' ==="
  XLA_FLAGS="${2}" JAX_ENABLE_X64=0 srun --ntasks=128 --ntasks-per-node=4 \
      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices 128 --mode strong \
      --n-lat 2048 --n-lon 4096 --nlev 26 --steps 12 --warmup 3 \
      --out "$OUTDIR/$1.jsonl" || { echo "$1 FAILED"; rc=1; }
}
run_arm A_default ""
run_arm B_combine "--xla_gpu_collective_permute_combine_threshold_bytes=8388608"
run_arm C_combine_pipelined "--xla_gpu_collective_permute_combine_threshold_bytes=8388608 --xla_gpu_enable_pipelined_p2p=true"
echo "=== RESULTS ==="
for T in A_default B_combine C_combine_pipelined; do
  "$PY" -c "
import json,math,sys
try:
    d=json.loads(open('$OUTDIR/$T.jsonl').readline())
    ms=d['steady_median_ms']; assert math.isfinite(ms) and ms>0
except Exception as e:
    print('$T: MISSING/INVALID ->', e); sys.exit(1)
print(f'$T: {ms:8.3f} ms')" || { echo "$T receipt invalid"; rc=1; }
done
echo "DONE rc=$rc"; exit $rc

codex
The section is not shippable unchanged. The census values reproduce, but the @128 f32 bound is arithmetically wrong.

1. **Bound arithmetic — blocker.** With the stated exact f32 payload (18,541,632 B), 41 × 26.3 µs, 23.5 GB/s, and one reduction:

   | row | correct t_bound | correct measured/bound |
   |---|---:|---:|
   | @64 f32 | 2.8633 ms | 2.3513 |
   | @128 f32 | **1.8936 ms** | **2.9450** |
   | @128 f64 | 2.9993 ms | 3.2012 |

   The published 1.848 ms / 3.02 for @128 f32 cannot arise from those inputs. `rank_imbalance` is diagnostic-only in [calibrated_bound](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/bench/metadata.py:754), not added to `t_bound`.

2. **Census probe — values reproduce, robustness needs a guard.** I reran the probe: nd=8, 16, and 128 all give 41 CP + 1 all-reduce; 1024→2048 gives 4,635,408→9,263,888 B (0.074% from exact doubling). Update the text to include nd=128; 8/16 alone supports an expectation, not a 128-device claim.

   The regex in [the probe](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/tmp/_probe_latlon_halo_census.py:65) does not double-count `-done`: it matches bare CP and `-start`, not `-done`. Result shape is valid for synchronous CP because it is shape-preserving. But the parser should compile once, assert `cp_ops_with_shape == census["collective_permute"]`, and retain per-op dtype/shape records; async tuple layouts are otherwise untested.

   The f64 “×2” is numerically harmless but not actually measured by the current probe. After steady-state x64 warm-up I find 9,270,800 B at n_lon=1024: four 4-byte scalar CPs remain f32, versus an exact 2×f32 value of 9,270,816 B. Make this a probe mode rather than an extrapolation.

3. **Claims/caveats.** “~2.4–3.2× above this *model*” remains true after correction (actual range 2.35–3.20), but replace “fabric+compute floor” with “fabric+compute model.” The pole-tile compute bias is not proven high/conservative; label its direction PLAUSIBLE or remove it, especially because f64 is compute-dominated. Also state that the 8-process/2-node IB calibration is being extrapolated to a 128-process/32-node communicator.

   The cited work supports the general small/medium GPU-collective latency concern, not this XLA schedule’s mechanism; the current PLAUSIBLE label is appropriate. [arXiv:2607.16100](https://arxiv.org/abs/2607.16100)

4. **A/B falsifiability is too strong.** The submitted script must record the actual GPU post-pass CP count/bytes for each arm; otherwise a ≤2% result only refutes this threshold/implementation, not “combinable-CP count.” Add repeat or A–B–A bracketing. C confounds combining and pipelined p2p, which is fine as an interaction arm but cannot attribute either. Finally, “ocean null was reduction-dominated” overstates the contrast: its documented halo and reduction model terms are roughly equal. Say the ocean has a different implicit-PCG/dependency mix, so its null is not predictive of atmosphere.

VERDICT: FIX-FIRST
tokens used
183,919
The section is not shippable unchanged. The census values reproduce, but the @128 f32 bound is arithmetically wrong.

1. **Bound arithmetic — blocker.** With the stated exact f32 payload (18,541,632 B), 41 × 26.3 µs, 23.5 GB/s, and one reduction:

   | row | correct t_bound | correct measured/bound |
   |---|---:|---:|
   | @64 f32 | 2.8633 ms | 2.3513 |
   | @128 f32 | **1.8936 ms** | **2.9450** |
   | @128 f64 | 2.9993 ms | 3.2012 |

   The published 1.848 ms / 3.02 for @128 f32 cannot arise from those inputs. `rank_imbalance` is diagnostic-only in [calibrated_bound](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/bench/metadata.py:754), not added to `t_bound`.

2. **Census probe — values reproduce, robustness needs a guard.** I reran the probe: nd=8, 16, and 128 all give 41 CP + 1 all-reduce; 1024→2048 gives 4,635,408→9,263,888 B (0.074% from exact doubling). Update the text to include nd=128; 8/16 alone supports an expectation, not a 128-device claim.

   The regex in [the probe](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/tmp/_probe_latlon_halo_census.py:65) does not double-count `-done`: it matches bare CP and `-start`, not `-done`. Result shape is valid for synchronous CP because it is shape-preserving. But the parser should compile once, assert `cp_ops_with_shape == census["collective_permute"]`, and retain per-op dtype/shape records; async tuple layouts are otherwise untested.

   The f64 “×2” is numerically harmless but not actually measured by the current probe. After steady-state x64 warm-up I find 9,270,800 B at n_lon=1024: four 4-byte scalar CPs remain f32, versus an exact 2×f32 value of 9,270,816 B. Make this a probe mode rather than an extrapolation.

3. **Claims/caveats.** “~2.4–3.2× above this *model*” remains true after correction (actual range 2.35–3.20), but replace “fabric+compute floor” with “fabric+compute model.” The pole-tile compute bias is not proven high/conservative; label its direction PLAUSIBLE or remove it, especially because f64 is compute-dominated. Also state that the 8-process/2-node IB calibration is being extrapolated to a 128-process/32-node communicator.

   The cited work supports the general small/medium GPU-collective latency concern, not this XLA schedule’s mechanism; the current PLAUSIBLE label is appropriate. [arXiv:2607.16100](https://arxiv.org/abs/2607.16100)

4. **A/B falsifiability is too strong.** The submitted script must record the actual GPU post-pass CP count/bytes for each arm; otherwise a ≤2% result only refutes this threshold/implementation, not “combinable-CP count.” Add repeat or A–B–A bracketing. C confounds combining and pipelined p2p, which is fine as an interaction arm but cannot attribute either. Finally, “ocean null was reduction-dominated” overstates the contrast: its documented halo and reduction model terms are roughly equal. Say the ocean has a different implicit-PCG/dependency mix, so its null is not predictive of atmosphere.

VERDICT: FIX-FIRST
