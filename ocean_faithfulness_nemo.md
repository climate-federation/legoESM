# Ocean faithfulness vs NEMO — all-grid correction tracker

**Goal:** correct ALL legoESM ocean grids (tripole/eORCA1, latlon, cubed_sphere,
mpas, spectral) so each gives a faithful comparison to NEMO ORCA1 (CORE-II NYF).
Review every code change with `/codex:adversarial-review`. Shrink this file every
10 iters. (Detailed history: `OMIP_faithful.md`.)

**Branch:** `omip-faithful-nemo-comparison`. Completion promise DONE only when grids
genuinely match NEMO — far off; no false DONE.

## State (carried over)
- NEMO ORCA1 5-yr reference COMPLETE (T/S/SSH/MLD/U/V/ice). Identical CORE-II
  6-hourly forcing (`nyf.zarr`), eORCA1 mesh, WOA18 all staged.
- Pipeline built + proven: `scripts/run_omip_core2.py` (runner, tripole + latlon_bathy),
  `build_core2_nyf_zarr.py`, `compare_omip_nemo.py` (regrid + SST/SSS score).
- Applicator fixes (39 tests pass): tripole branch, wind-stress sign (−tau), 6-hourly
  flux, true periodic NN, in-step forcing (`compute_omip2_surface_forcing`).

## THE BLOCKER (root cause, rigorously diagnosed)
legoESM ocean blows up from realistic (WOA) stratification — NOT topo/IC/convection/
dissipation/dt/**KE-gradient-scheme** (ruled out via flat-bottom, static-stability,
~19 experiments). KE scheme ruled out iter 2 (job 8106193): centered AND hollingsworth
both go non-finite by day 0.5 on tripole AND latlon-bathy → NOT the Hollingsworth
instability, despite the plausible centered-KE+AL81-PV inconsistency hypothesis.
Mechanism: cold-start geostrophic adjustment from rest (u=0) overshoots to ~5-10 m/s
→ nonlinear advective feedback (u·∇u) → blowup, amplified at the **equator (f→0)**.
- **FIX #1 DONE (kept):** WOA-IC flood-fill — NEMO mask had ~13% ocean cells WOA lacks
  data for (S=0 → 40-PSU spurious gradients → 12 m/s step-1 PGF). Now filled from nearest
  valid column (step-1 |u| 12→0.8, mean SSS 29.7→34.1).
- **In progress:** spin-up Rayleigh velocity drag (damp the cold-start overshoot during a
  nudged spin-up, then release) — test 8100129 pending. If insufficient → thermal-wind-
  balanced init.

## Per-grid status
| grid | runs stable (realistic IC)? | comparison |
|---|---|---|
| tripole/eORCA1 | NO (cold-start blowup; flood-fill + drag in progress) | pending |
| latlon_bathy | NO (same; regridded NEMO bathy) | pending |
| cubed_sphere | untested w/ CORE-II | — (applicator supports) |
| mpas | untested w/ CORE-II | — (applicator supports) |
| spectral | applicator unsupported | TODO |

## Next
1. **Per-term tendency instrumentation** of `model.step` over steps 1-10 (WOA cold-start):
   log max|tendency| per term (Coriolis/PV-flux, KE-grad, PGF, vert-adv, viscosity) +
   the lat/lon of the max — find WHICH term grows first + WHERE. Blowup is grid- AND
   KE-scheme-independent ⇒ shared `LatLonCGridOceanModel` baroclinic dynamics. This is
   the un-done diagnostic; do it before the next blind lever.
2. Un-ruled-out core levers (after #1 points the way): PGF density-Jacobian / EOS audit;
   barotropic↔baroclinic split coupling; baroclinic-mode discretisation.
3. Drag spin-up (job 8100129) is a symptom-treatment fallback if a real fix stalls.
4. Once a grid runs stable + free → compare to NEMO (SST/SSS, then ACC/AMOC/MOC/MLD).
5. All-grid audit (cubed_sphere/mpas/spectral) in flight (workflow wnz1vbd2y) — fold in.
6. codex-adversarial-review each change.

### Deferred (only if KE scheme ever matters)
Hollingsworth KE stencil (`ocean_pe_latlon_cgrid.py` ~1010-1033) widens to j±1 with
edge-replication wall halos; on the TRIPOLE it ignores the north-fold permutation/sign
(codex high finding, iter 2). A fold-aware KE halo + active-tripole regression test is
the prerequisite to ever making `ke_gradient_scheme="hollingsworth"` the tripole default.
A/B knob added: `run_omip_core2.py --ke-gradient-scheme {centered,hollingsworth}`.

## Iteration log
- **iter 1:** new loop. Drag-spinup test 8100129 queued (GPU busy). Created this tracker.
  codex-adversarial-review of dycore changes = needs-attention, 2 valid findings, both FIXED:
  (1) [high] flood-fill copied donor columns without reapplying the RECEIVER bathymetry deep-fill
  → below-seafloor T/S bias; now reapplies the deep-fill (z_cen > H_bathy → deep-ocean fill).
  (2) [med] NN forcing-sampler cache key omitted dst_lon.sum() → same-shape grids could collide;
  key now signs full src+dst lat/lon checksums. Drag test 8100129 (PD) will run the fixed code.
- **iter 2 (KE-gradient scheme RULED OUT — empirical):** Hypothesis: the realistic-IC blowup
  is the Hollingsworth-Kållberg instability — the default `ke_gradient_scheme="centered"` KE
  gradient pairs inconsistently with the AL81 12-point PV-flux Coriolis term; the realistic
  DINO experiment uses `"hollingsworth"` (#263) and KE-scheme was NEVER among the ~16 prior
  levers. Built A/B knob `run_omip_core2.py --ke-gradient-scheme` (codex-clean, 4 rounds).
  **A/B job 8106193 (A40): tripole centered (control), tripole hollingsworth, latlon-bathy
  hollingsworth — ALL THREE finite at step 0 (SST~13°C, |u|=0) then non-finite by step 72
  (day 0.5). Hollingsworth ≡ centered. KE scheme is NOT the blocker.** Negative result, but a
  real lever crossed off (19 total) + reusable A/B infra. Codex flagged [high]: the
  hollingsworth KE stencil lacks a fold-aware north halo → reverted both config DEFAULTS
  (kept centered), knob-only diff. Refined target: blowup is grid- AND KE-independent ⇒
  shared baroclinic dynamics; next = per-term tendency instrumentation (steps 1-10).
