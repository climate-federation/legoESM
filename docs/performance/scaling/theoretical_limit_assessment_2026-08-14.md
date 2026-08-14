# Distance to the theoretical line — dual-review assessment (2026-08-14)

Question: what stands between the measured numbers and ideal strong
scaling for the MPAS and lat-lon lanes at 64–256+ A100s. Two independent
reviews of the same brief (GLM-5.2 and codex/gpt-5.6, both with the full
measured budget and the refuted-lever list), synthesised here. User
directive: 3-D decomposition (adding the LEVEL axis) is to be assessed
first; the 2-D lat-lon tile lever is deprioritised.

## The measured starting point

MPAS subdiv-9 @ 64 (job 26942819): full 6.820 ms =
kernel-no-halo 3.460 + wire 1.815 + skew 1.230 + staging 0.315.
Same arms at 4 devices, same per-device load: 2.930 / 2.780 / 2.725.
Serial single-GPU at that load: 2.000 ms => ideal@64 ≈ 2.0, so 3.4x off.
Lat-lon LL2048 @ 128: 4.527 ms, 13 collective-permutes/step, 1-D bands.

## Where the reviewers AGREE (act on these)

1. **The budget's two waiting terms are partly the same thing, and both
   are mis-attributed until per-rank timing exists.** The arms time
   wall/max over ranks, so inter-rank arrival variance lands in
   whichever bucket is being measured. Both name the same first
   measurement: per-rank device-event timestamps on the existing arms,
   plus schedule-aware partner matching in the existing JAX-trace
   analyzer (`scripts/bench/analyze_jax_trace_gaps.py` — nsys is blind
   to the collectives on this stack, the JAX profiler is not).
   ~0.5 week; re-attributes up to 1.2 ms before anything is built.

2. **The +0.735 ms no-halo growth 4→64 is mostly explicable, not
   mysterious.** Codex, from the code: the wide-halo kernel computes the
   FULL padded local region at every RK evaluation and masks afterwards
   (`sharded_dynamics.py` wide kernel), and every rank pads to the
   WORST shard's extents — at 64 the padded extent is imposed on all.
   Predicted split ~0.35–0.45 redundant deep-ring compute +
   0.10–0.20 mass-fix allreduce + 0.10–0.25 fusion/layout drift. The
   queued fix-vs-nofix arm prices the middle term; a "compact the RHS to
   the RK-valid ring" change prices the first (predicted −0.30 ms,
   CONFIRM kernel-no-halo ≤ 3.20 ms).

3. **Rank→GPU/NIC binding is unexamined and cheap.** Collective arrival
   skew from topology/affinity was never controlled (`--gpu-bind=none`
   everywhere). GLM: swap-ballast test discriminates topology from
   compute variance. Codex: predicted 0.60 ms, 1–2 days, CONFIRM full
   step ≤ 6.42 ms.

4. **Reduced-precision halos are NOT a transparent optimisation.**
   Codex: f16 spacing at 100 kPa is ~64 Pa; bf16 mantissa worse; casting
   changes the model — permissible only as a separately named lossy mode
   behind a multi-day conservation + spectra gate. GLM: field-selective
   (tracers only) is the defensible subset, ~3 weeks with validation.
   Wire prize if the full gate passes: ~0.9 ms.

5. **Per-FIELD halo depth is not the lever** (the RHS couples the full
   state; different depths reintroduce epochs). The worthwhile variant
   is per-EVALUATION compact compute — the same item as (2).

6. **The >256 structural bet is the K-step swept (time-blocked) halo**:
   exchange once at depth K·evals·3, run K whole steps locally on a
   shrinking region, keep only the per-step scalar conservation
   reduction. It attacks collective count and arrival latency — the
   terms that grow with N — and MPAS already has the K=1 ancestor (the
   wide halo). Gate before building: HBM fit at the target subdiv (the
   depth-9 fill already OOMs at subdiv-10@128 — memory is the risk),
   and a real K=2 A/B saving ≥ 20 %.

## Where they DISAGREE (resolve by measurement, not argument)

* **Lat-lon 2-D tiles at 128.** GLM: switch now, ~2.0 ms prize, pole
  all_gather is 1–2 MB and irrelevant. Codex, from the code: the packed
  stage-entry path REFUSES a 2-D mesh (`latlon_spmd.py`
  `packed_exchange_mesh`), both staggerings must be reconstructed, and
  at 128 the 2-D step likely LOSES (−0.2 ms prior); promote only on a
  full-step A/B ≤ 3.85 ms, and only even `p_lon` (the fold is then a
  partner ppermute, not an all_gather). Deprioritised per user
  directive regardless; the A/B is the eventual >256 gate.

* **bf16 halo scope** (tracers-only vs not-at-all until gated): both
  reduce to "behind a validation gate"; the gate defines the scope.

## 3-D (level) decomposition — the user-directed question

Prior art in-repo: the SPECTRAL dycore anti-scales under level sharding
(measured 0.74x at 4 devices, T85) because the semi-implicit solve
couples all levels. The finite-volume lanes were never censused. The
offline gate (`scripts/tmp/_probe_level_shard_census.py`) shards the
LEVEL axis of the serial step across virtual CPU devices via GSPMD and
counts the compiled collectives:

* O(1)–O(10) collectives/step → a 3-D (horizontal × level) mesh is
  worth a GPU measurement;
* O(nlev)+ → every column operator (hydrostatic cumsum, vertical
  advection, implicit vertical diffusion) serialises across level
  shards and 3-D is dead for these lanes, as it is for spectral.

RESULT: pending (compiling at time of writing; recorded in the
follow-up commit).

Physics of the expectation: both lanes' vertical operators are
per-column recurrences (cumsum for hydrostatic pressure/geopotential,
tridiagonal solves for implicit vertical mixing). A recurrence across a
sharded axis lowers to a sequential chain of per-shard segments — the
same structure that killed spectral. The census is the check that GSPMD
does not find something cleverer.

## Ranked plan (ms recovered per week, both reviews merged)

| # | lever | predicted | cost | gate |
|---|---|---|---|---|
| 1 | per-rank timestamps + partner matching (re-attribute skew) | 0–1.2 reattributed | 0.5 wk | spread ≥ 0.8 ms confirms skew real |
| 2 | rank→GPU/NIC binding | 0.6 | 0.3 wk | step ≤ 6.42 ms |
| 3 | mass-fix pricing (queued job 26952029) | 0.1–0.7 | 0 (queued) | fix−nofix ∈ [0.55,0.95] |
| 4 | compact wide-halo RHS to the valid ring | 0.3 | 1 wk | kernel-no-halo ≤ 3.20 ms |
| 5 | 3-D level shard census → GPU arm only if census passes | unknown | 0.2 wk gate | ≤ O(10) collectives |
| 6 | bf16 tracer halos behind a conservation gate | 0.6–0.9 | 3 wk | codex's two-stage gate |
| 7 | K-step swept halo (>256 bet) | ~0.8 @256 | 4–6 wk | HBM fit + K=2 ≥ 20 % |
| 8 | lat-lon 2-D tile A/B (deferred per directive) | −0.2…+2.0 | 1 wk | full step ≤ 3.85 ms @128 |

Kernel roofline work (single-GPU 6x off HBM) is the largest absolute
pool (~1.5 ms at 64) and the worst rate (~0.25 ms/wk, codex and GLM
agree); it does not move the strong-scaling RATIO and queues behind
everything above.
