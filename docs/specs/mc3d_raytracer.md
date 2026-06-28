# 3D Monte-Carlo Ray-Tracing Radiation for LES/CRM (`scheme="mc3d"`)

Status: **Phase 1 + Phase 2 complete (codex-clean); Phase 2b/3/4/5 pending**. Port of the microhh `rte-rrtmgp-cpp`
CUDA ray tracer (`src_cuda_rt/Raytracer.cu`, `Raytracer_bw.cu`, `Rte_lw_rt.cu`)
into pure JAX as a new, selectable radiation scheme for the doubly-periodic
plane LES/CRM dycores. Reuses legoESM's existing RRTMGP correlated-k optics
(per-g-point `optical_depth`, `ssa`, `asymmetry_factor`) — the ray tracer
replaces only the *spatial solver* (column two-stream → 3D Monte-Carlo
transport), not the optics.

## Motivation

legoESM radiation today is the Independent Column Approximation (ICA): each
column is solved by a plane-parallel two-stream solver, no horizontal photon
transport (`radiation/integration.py:697`). For LES/CRM at O(10–100 m)
resolution, 3D radiative effects — cloud-side illumination, shadows, horizontal
photon leakage, entrapment — are first-order on cloud heating/dissipation and
are entirely absent under ICA. The microhh forward Monte-Carlo tracer is the
reference solution for these effects.

## Scope (locked with user)

- **Differentiability:** Phase A forward-only, AD-gated (treated like
  `global_max_mpi` — a diagnostic forcing, kept out of losses). Phase B: train a
  differentiable neural emulator against the MC tracer.
- **Backend:** pure JAX (cross-backend CPU/GPU/Metal, pure-pytree, `lax`
  control flow). No CUDA FFI.
- **Spectral:** SW + LW.
- **Geometry:** doubly-periodic plane (`spectral_les_plane`,
  `compressible_euler_plane`, `pseudo_incompressible_plane`). NOT cubed-sphere /
  global — periodic horizontal BC gives clean photon wraparound.

## Algorithm (mirrors microhh)

Forward photon Monte-Carlo with **null-collision (Woodcock / delta) tracking**:

1. **Optics per g-point.** From RRTMGP optics get `optical_depth(nx,ny,nz)`,
   `ssa`, `g`. Convert to extinction `k_ext = optical_depth / dz`,
   scattering `k_sca = ssa * k_ext`. Loop g-points serially → only one
   g-point's 3D field resident at a time (the key memory lever; see below).
2. **Majorant grid.** Coarse `k_null` grid = per-coarse-cell max `k_ext`
   (floored at `1e-3`), built by `create_knull_grid`. Enables delta tracking:
   sample free path with majorant, accept a real collision with prob
   `k_ext/k_null`, else null-scatter (continue straight).
3. **Photon source.**
   - **SW:** collimated solar beam injected at TOA over each column, direction
     from `(mu0, azimuth)`; weight = TOA flux per photon.
   - **LW:** volumetric thermal emission ∝ `k_abs * B(T)` per cell + surface
     emission ∝ `(1-albedo_lw) * B(T_sfc)`. Photons sampled proportional to
     local emission (importance sampling on the Planck source).
4. **Walk** (`lax.while_loop` per photon, vmapped/scanned over a photon batch):
   free-path sample → advance with periodic horizontal wrap, reflect/exit at
   TOA, interact at surface → at a real collision: absorb (Russian roulette or
   weight reduction by `ssa`) or scatter (sample new direction from phase
   function). Terminate on absorption, surface absorption, TOA escape, or weight
   floor.
5. **Phase functions:** Rayleigh (gas `k_sca_gas`), Henyey–Greenstein for
   cloud/aerosol (asymmetry `g`), surface BRDF = Lambertian albedo (SW
   direct/diffuse split tracked separately, as in microhh).
6. **Tally.** Scatter-add photon events to flux/count grids:
   `flux_sfc_dir/dif/up`, `flux_tod_dn/up`, 3D `flux_abs` (→ heating rate).
   `count_to_flux` normalizes counts → W/m². 3D absorption → `dθ/dt`.

### RNG

microhh uses Sobol QRNG (cuRAND). Phase A uses JAX counter-based `threefry`
(`jax.random`) — one key per photon, split per event. QRNG variance reduction is
a Phase-A.2 optimization (low-discrepancy via `jax.random` is not built in;
defer). Deterministic + reproducible across backends, which Sobol-via-cuRAND is
not.

## Parallelization & GPU memory (the user's flagged concern)

**Memory is NOT the bottleneck if structured per-g-point.** For a 256×256×256
LES grid:

| Array | Size | Note |
|---|---|---|
| `k_ext`/`k_sca`/`g`, **one g-point** | ~0.3 GB | g-point loop ⇒ NOT ×112 |
| Photon live state, P=128 | ~0.27 GB | `lax.scan`-chunk batches if tight |
| Photon live state, P=1024 | ~2 GB | chunkable to any ceiling |
| 3D flux/count tally | ~67 MB | grid-sized |

The g-point serial loop is exactly microhh's trick: never materialize the full
spectral optical field. Photon batches stream via `lax.scan` so peak memory is a
free parameter independent of photon count.

**Multi-GPU = embarrassingly parallel by sharding photons, NOT the domain:**

1. Replicate the per-g-point optical field on every rank (~0.3 GB each — cheap).
2. Shard photon batches across ranks (distinct RNG key offset per rank).
3. `allreduce-SUM` the flux/count tally grids. SUM allreduce is the one AD-safe
   reduction (`global_sum_mpi`), so this survives the Phase-B differentiable
   path.

This avoids cross-subdomain **photon migration** (the genuinely hard distributed
ray-tracing problem) entirely, and scales near-linearly. Variance ~ 1/√(N·ranks).

**Cost control:** 3D RT is called every `radiation_call_interval` steps (not
every step); photon count + majorant-grid coarseness traded against variance.
XLA runs the divergent `while_loop` with all lanes masked (no CUDA warp
compaction) → expect ~5–50× slower per-photon than microhh, amortized by the
call interval and photon-count tuning.

## Module layout

```
radiation/mc3d/
  __init__.py
  config.py            # MC3DRadiationConfig (NamedTuple) + __param_spec__
  knull_grid.py        # build_majorant_grid()
  sampling.py          # rayleigh/HG phase fns, free-path, surface BRDF
  photon_walk.py       # core lax.while_loop single-photon transport
  raytracer_sw.py      # SW driver: source -> batched walk -> tally
  raytracer_lw.py      # LW driver: emission source -> walk -> tally
  tally.py             # count->flux, scatter-add, MPI allreduce-SUM
  parallel.py          # photon-shard / replicate-optics helpers
```

Wired into `make_radiation_physics` via `scheme="mc3d"`, plane-only:
`_make_plane_radiation` gains an `mc3d` branch (unknown scheme already
`raise ValueError` at `integration.py:156`). `MC3DRadiationConfig` carried on
`RadiationConfig.mc3d`.

## Parameter hygiene

- `photons_per_pixel`, `n_knull_coarsen`, `max_scatter_events` are **int counts**
  → module constants / config ints, **never trainable** (loop-counts-never-
  trainable rule; ints are not `__param_spec__`-eligible).
- `knull_floor=1e-3`, weight floors → `# coeff-ok:` numerics one-offs or named
  module constants with provenance.
- Surface albedo, LW emissivity = already in `RadiationConfig` /
  scheme config, tier-1 tunable.
- No re-derived saturation/constants — `from legoesm import constants`,
  Planck from existing `optics_lib.compute_planck_sources`.

## Test plan (truth tiers first, oracle last)

1. **Analytic — homogeneous slab.** Uniform `k_ext`, no scattering: MC mean
   transmittance → Beer–Lambert `exp(-τ/μ0)` within MC error (1/√N). Pure
   absorption energy balance.
2. **Conservation (energy).** `incident = reflected + absorbed + transmitted`
   to MC tolerance, per g-point and broadband. Hard gate.
3. **Two-stream limit.** Horizontally-uniform cloud → MC 3D fluxes converge to
   the existing column two-stream solver (`rte/two_stream.py`) within MC error.
   Ties the new solver to the validated 1D path.
4. **Convergence.** Error ∝ 1/√N verified across photon counts.
5. **Equivariance.** Horizontal shift of the optical field shifts the flux field
   identically (periodic-BC equivariance) to MC tolerance.
6. **Parallel.** Single-rank vs photon-sharded multi-rank agree (same total
   photons, allreduce-SUM) within MC error; unsharded vs sharded.
7. **Visual.** Cloud-field shadow/side-illumination PNG vs ICA — the 3D effect
   must be visibly present (norms necessary, not sufficient — CLAUDE.md visual
   gate).
8. **Oracle (tier 3, last).** Match microhh on a shared cloud field, trusted
   only after tiers 1–6 pass.

## Phased delivery (each PR: codex adversarial-review loop to clean)

- **Phase 1 [DONE]** — core SW: `photon_walk`, `knull_grid`, `sampling`,
  `raytracer_sw`, `tally`, `config`; tests 1,2,4 (slab, conservation,
  convergence). Single-rank. Codex-clean (3 rounds).
- **Phase 2 [DONE]** — `solve_sw_spectral` (g-point accumulation) + `SWFluxResult`;
  `plane_adapter` (geometry/orientation bridge, gray SW optics, heating
  conversion); `scheme="mc3d"` wired into `_make_plane_radiation` (3D-MC SW +
  gray LW) with mc3d-only-plane + instantaneous-zenith guards; tests 3,5,7
  (two-stream cross-check, equivariance, visual shadow) + spectral budget +
  end-to-end dispatch. Codex-clean (2 rounds). NOTE: live optics = gray
  (ssa=g=0 ⇒ Beer-Lambert anchor); scattering transport validated via prescribed
  cloud fields. Perf: eager x64 CPU is slow for cloudy fields (scalar global
  majorant ⇒ tiny null steps in clear air) — motivates Phase 1.1 + Phase 4.
- **Phase 1.1 [DONE]** — coarse `k_null` majorant grid (`build_majorant_grid` +
  `MajorantGrid`): per-block-max majorant + DDA (carried coarse-cell indices,
  no position re-floor) decomposition Woodcock tracking. `knull_coarsen_xy/_z`
  config knobs (0 = global scalar = unchanged default). Pure speedup, unbiased
  (verified coarse-vs-scalar agree for vertical + slant/scattering/seam cloud
  fields). Codex-clean (5 rounds): fixed DDA periodic-seam frame mismatch,
  tiny-component face-skip, corner-tie zero-step, on-face 0/0 NaN (AD-safe
  denominators).
- **Phase 2b [DONE]** — RRTMGP-spectral optics provider: `two_stream.
  compute_sw_optical_props_gpt` (extracted, shared with `solve_sw`) +
  `compute_sw_optical_field`; `RRTMGP.solve_columns(sw_optical_field_only=True)`
  reuses the validated state-build prologue in place (default path byte-identical
  — full RRTMGP regression suite passes) and returns per-g-point (tau,ssa,g) +
  solar weights; `_mc3d_plane_heating` feeds them (with cloud optics) to the MC
  tracer. mc3d builds the RRTMGP solver (gray fallback if tables absent); cloud
  gate extended to mc3d. Codex-clean (2 rounds).
- **Phase 3 [DONE]** — LW forward thermal-emission MC (`raytracer_lw`,
  `solve_lw_monochromatic`): volume (4π k B dV) + surface (π ε B dA) emission
  sampled by categorical, reuses the walk with ssa=0 / albedo=1-ε, net =
  ε·(absorbed-emitted). `compute_plane_lw_heating` adapter; mc3d LW = 3D-MC
  (gray Planck/optics) replacing the gray two-stream column LW. Tests: energy
  conservation, analytic optically-thin isothermal cooling (-4kσT⁴dz), OLR sign,
  all-cold no-NaN, maxiter-unresolved budget, determinism. Codex-clean (3
  rounds): OLR excludes unresolved max-iter photons (honest budget term), float32
  e_tot=0 NaN guard.
- **Phase 3b [DONE]** — RRTMGP-spectral LW optics: `two_stream.
  compute_lw_optical_field` (per-g-point absorption OD + Planck radiance + LW
  aerosol absorption); `solve_columns(lw_optical_field_only=True)` (byte-
  identical default — RRTMGP regression passes); `raytracer_lw.solve_lw_spectral`;
  `compute_plane_lw_heating_spectral`; mc3d LW = RRTMGP-spectral (cloud-aware)
  when the solver is present, gray LW fallback otherwise. Codex-clean: Planck
  source is RADIANCE B (pass straight, no /pi — certified by the OLR-scale test
  mc3d 365 vs two-stream 275 W/m^2, ratio 1.32 = O(1), not pi/pi^2); LW aerosol
  forwarded. Tests: spectral conservation, RRTMGP LW extraction, OLR-scale.
- **Phase 3c [DONE]** — linear-in-tau in-cell LW emission: `solve_lw_monochromatic`
  takes optional per-cell lower/upper-face Planck radiance; cell weight = face
  mean, emission depth sampled by the inverse-CDF of the linear profile so
  escaping photons carry the cooler cell-top temperature (removes the cell-center
  over-emission). Face Planck threaded RRTMGP `planck_src_bottom/top` ->
  `solve_columns` (5-tuple) -> `compute_plane_lw_heating_spectral` -> `solve_lw_
  spectral`; orientation (lower-z face) preserved through both z-reversals.
  Cell-center stays the default (faces None). Codex-clean (6/6 PASS). Also fixed
  an orientation bug in the OLR-scale test (now via the production adapter).
- **Phase 4 [DONE]** — `parallel.py`: `solve_sw_sharded` (fold-accumulate photon
  shards, real memory lever), `average_results`, `pmean_result` (pmap/shard_map,
  AD-safe); `scripts/bench/bench_mc3d_raytracer.py`. Tests: sharded==single,
  variance, AD-safe combine, pmean-under-pmap, bench smoke. Codex-reviewed (no
  correctness bugs; 2 quality fixes: fold not stack, bench touches all leaves).
- **BOMEX LES integration [DONE]** — `tests/unit/test_mc3d_bomex.py`: mc3d runs
  end-to-end through `make_radiation_physics(model_type="plane")` on a BOMEX
  (Siebesma 2003) shallow-cumulus plane state (analytic θ_l/q_t + broken q_c,
  RRTMGP SW cloud optics) — finite, conserves, no tracer movement; broken-cloud
  surface shadow. Full multi-hour run needs an external gSAM CASES/BOMEX deck.
- **Explicit Rayleigh + QRNG [DONE]** — `sampling.scatter_direction_mixed`
  (explicit Rayleigh 1+mu^2 gas phase split from HG cloud by the per-cell
  Rayleigh fraction) + `qrng.py` opt-in Halton low-discrepancy launch
  (`config.use_qrng`). Both codex-clean. Mie-CDF cloud phase remains blocked
  (microhh external Mie tables); HG(g_cloud) is the flux-faithful stand-in.
- **Oracle fidelity [DOCUMENTED]** — `docs/specs/mc3d_oracle_fidelity.md`: algo-
  by-algo map to microhh `src_cuda_rt` + truth-tier evidence + measured GPU
  speedup. Algorithm-fidelity COMPLETE; remaining = data-blocked (Mie tables),
  CUDA head-to-head, or differentiability. Truth tiers (which outrank oracle-
  matching) cleared.
- **Phase 5 [IN PROGRESS]** — differentiable 3D-CNN (U-Net) emulator of the MC
  SW transport: `mc3d/emulator.py` (`UNet3D` + `pack_inputs`, softplus head,
  jax.grad-clean), `scripts/data/gen_mc3d_emulator_data.py` (random broken-cumulus
  optical fields + Mie -> MC abs_frac targets), `scripts/experiment/
  train_mc3d_emulator.py` (`create_optimizer` warmup+cosine+clip, train/val).
  Tests: forward/non-neg, differentiable, training-reduces-loss. Codex-clean
  (odd-kernel / val-split / epochs guards). Substantial-dataset training run
  pending (CPU data-gen). Maps 3D optical field -> 3D heating (per user).
- **Phase 5 (orig)** — neural emulator trained against MC tracer (differentiable
  Phase B); separate spec.
