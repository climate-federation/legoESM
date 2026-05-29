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
| tripole/eORCA1 | partial-cell coord lands the worst fix; vertadv-amplified equatorial residual | pending |
| latlon_bathy | same dynamics path; same residual | pending |
| cubed_sphere | untested w/ CORE-II | — (applicator supports) |
| mpas | untested w/ CORE-II | — (applicator supports) |
| spectral | applicator unsupported | TODO |

## All-grid audit (workflow wnz1vbd2y, 7 agents) — "correct all grid types" map
- **tripole/latlon (LatLonCGridOceanModel):** vertical-coord TYPE fixed (partial-cell). Residual
  = vertadv amplifier + equatorial PGF seed (above). Coriolis from geographic lat = correct.
  **Follow-up [accuracy, non-blocking]:** the HW KE branch (ocean_pe_latlon_cgrid.py:1015-1018)
  fills j±1 u-neighbour by edge-replication — WRONG at the active tripole north fold (should be
  `vector_sign_u·u[-1:,perm_T]`). Affects `run_tripole_20yr.py:83` + any HW-on-tripole run
  (one-row Arctic seam, NOT the equatorial blowup). Fix = fold-aware halo (mirror
  `pad_ns_vector_u`/`_fold_row`) + synthetic-fold unit test (current test is no-fold) + VISUAL
  verify the fold row. Centered KE is fold-safe (no j±1 reach).
- **cubed_sphere (OceanModel/ocean_pe_cdgrid):** Coriolis correct (geographic lat, edge-synced).
  Momentum HK-immune (vorticity-from-circulation). **[high] documented face-edge PGF instability**
  (NaN ~2.2 d at cube-edge cells) currently MASKED by A_h≥5e5/K_h≥5e6 + FC-Gram → 5-50× heavier
  diffusion ⇒ NOT physically comparable to NEMO. Structural fix = SMC03 + duogrid halo on T/S.
- **mpas (MPASOceanModel/TRiSK):** **[high] split-Coriolis inconsistency (#160):** planetary f
  zeroed in the PV flux (q=ζ/h only) + applied by separate barotropic/Matsuno ops → breaks the
  energy-conserving TRiSK identity; q-transport (h·u_total) inconsistent w/ continuity flux;
  default `pv_scheme='enstrophy'` non-EC. Fix = MOM6 full-PV q=(f+ζ)/h + continuity transport.
  Also: untested w/ CORE-II (run it).
- **spectral (spectral_ocean_pe):** **[high] OMIP applicator gap is TOTAL** — `SpectralOceanState`
  has no grid-space u/v/T or masks, so both `apply_omip2_surface_fluxes` and
  `compute_omip2_surface_forcing` raise. Self-flagged unsupported (Gibbs ringing #99). Fix =
  spectral surface-forcing path (synth top layer→Gaussian grid→air_sea_fluxes→curl/div→vor/div_hat).

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
  **KEY CLUE (drag run 8100129: nudge tau60 + flood-fill + Rayleigh drag tau1/120d,
  centered KE):** survives to ~day 90, NaN day 100; the unstable mode is MERIDIONAL v ≫ u,
  pinned at the EQUATOR (umax_lat −4 to −6°): max|v| 2.7(d10)→7.8(d20)→18.1(d30) while
  max|u| only 1.8→3.3. Grows monotonically DESPITE active spin-up drag ⇒ drag-immune
  equatorial-v mode. At f≈0 a meridional PGF is unbalanced → prime suspect KE_PGF_v
  (meridional pressure gradient). Per-term diag job 8106208 (WOA cold-start, no forcing)
  will confirm which dv_dt term drives the equatorial v.
- **iter 2 (PER-TERM tendency localisation — MECHANISM FOUND, job 8106208):** instrumented
  `model.tendencies_with_diagnostics` every step on the WOA cold-start. No-forcing and
  with-forcing arms are STEP-FOR-STEP IDENTICAL ⇒ **CORE-II forcing exonerated** (it is a
  pure IC/dynamics blowup). Mechanism, decisively:
  • **SEED (steps 1-2): equatorial meridional PGF** `KE_PGF_v ≈ 8.6e-3 m/s² @ +4.4°N` drives
    v from rest to ~5 m/s in ONE step (8.6e-3·dt600 = 5.2). That is ~900× a physical
    baroclinic-PGF estimate (~1e-5 m/s²) ⇒ the equatorial meridional PGF seed is SPURIOUSLY
    LARGE. At f≈0 nothing arrests it.
  • **AMPLIFIER (steps 3-6): vertical momentum advection** `vertadv_u/v @ −3.3°N`. The
    PGF-driven v converges meridionally → spurious w (~0.03 m/s, ~300× physical) → flux-form
    upwind `∂(w·u)/∂z` explodes: vertadv 2.25e-2→0.31→12→6.7e5 → NaN step 10.
  • Explains why **dt 600/300/150 ALL failed** (growing-flow feedback, not a fixed CFL) and
    why it is **viscosity/drag-immune**.
  ⇒ Root = the spurious equatorial meridional PGF; vertadv turns it into the blowup. NEXT:
  seed-localisation probe 8106261 (is KE_PGF_v at the SURFACE = IC/flood-fill artifact, or
  DEEP = partial-cell PGF over topography? and at a specific lon?) → then fix the PGF/IC seed
  (and/or make vertical momentum advection implicit/limited to kill the amplifier).
- **iter 2 (ROOT CAUSE — wrong vertical-coordinate TYPE, "correct grid types"):** seed probe
  8106261 put the spurious KE_PGF_v at **lat 4.4°N, lon 123.5°E, k=19 (bottom level)** — the
  Indonesian seas, deepest level, steepest equatorial bathymetry; the whole cascade stays at
  lon 123.5°E. PGF A/B 8106265 (adcroft vs smc03 on tripole) came back **BIT-IDENTICAL** ⇒
  the `pgf_scheme` switch is a NO-OP. Cause: the Adcroft/SMC03 partial-cell PGF correction in
  `ocean_pe_latlon_cgrid.py:1071` is gated `isinstance(z_coord, OceanPartialCellCoordinate)`,
  but the OMIP runner passes the plain `OceanZStarCoordinate` from `_create_setup`. That coord
  has `J=(eta+H_bathy)/H_max` → ALL levels uniformly stretched to the local depth = **sigma-
  like / terrain-following**, NOT NEMO's z-level-with-partial-steps. So (a) the PGF correction
  never fires AND (b) over steep equatorial topo the sigma-PGF error is huge at f≈0.
  **`run_omip.py`'s OWN main driver (run_omip_single ~L3181) ALREADY converts to
  `OceanPartialCellCoordinate` + thin-cell-snaps** — its comment literally describes this
  exact "day-13 equatorial PGF instability, f≈0, thin partial cell, blow up". The OMIP-faithful
  runner (`run_omip_core2`) BYPASSED that stable setup. This invalidates the prior "smc03
  tested, still blew up" lever — smc03 was never actually applied.
  **FIX (this loop's "correct grid types"): added `make_partial_cell()` + `--partial-cell` to
  run_omip_core2 (z-level partial steps + thin-cell snap, NEMO-faithful; activates the PGF
  correction).**
  • **Per-term efficacy (job 8106758): CONFIRMS the mechanism.** With `--partial-cell` the
    super-exponential vertadv runaway is GONE (finite through 16 steps vs NaN by step 10 on
    plain z*); the equatorial KE_PGF_v seed DECAYS (8.1e-3→3.5e-3) instead of running away;
    and `pgf_scheme` now actually matters (adcroft≠smc03 arms) — proving it was gated off.
  • **IC bug found + fixed (codex 3 rounds → approve):** `compute_woa_3d` was deep-filling
    ACTIVE bottom partial cells (init_ocean_from_woa's `|z_full_ref|>bathymetry` mask + the
    re-apply pass) → corrupted IC at the topographic-step region. Now: pass
    `bathymetry_depth=None` for partial-cell coords + mask only `~is_active`.
  • **RESIDUAL (watch):** per-term shows partial-cell removes the catastrophic equatorial
    blowup but a slower ~linear growth persists (mid-lat deep PGF, e.g. −36.3°N/−50.5°E South
    Atlantic slope, ~3.5e-3 m/s²) → contaminated forced run blew up ~day 2. Clean forced
    validation (job 8106781, --partial-cell, adcroft vs smc03, 30d, fine diag) PENDING:
    does the clean IC + (now-active) smc03 PGF give a stable physical multi-day forced run?
  Code committed (partial-cell coord + partial-cell-aware WOA IC, codex-clean).
- **iter 2 (partial-cell NECESSARY but NOT SUFFICIENT — forced equatorial residual):** clean
  forced validation 8106781 (--partial-cell, clean IC, adcroft AND smc03, WOA cold-start, no
  drag) **blew up by day 0.5-1: max|u|=227, max|v|=124 @ EQUATOR (−1.3°N)**, then NaN day 1.
  So partial-cell removed the catastrophic step-10 Indonesian-seas super-exponential runaway
  (per-term 8106758 confirmed) but a FORCED equatorial instability remains. Note: per-term
  NO-forcing partial-cell survived 16 steps (~16 m/s, mid-lat), but WITH forcing it explodes
  at the equator ⇒ forcing now implicated (was masked on plain z* when the Indonesian
  bottom-PGF dominated). NEXT: per-term WITH forcing + partial-cell (job below) to localise
  the equatorial term (residual PGF? vortcor? surface-forcing `phys`? barotropic-split?).
  Also retry the GRADUAL path (partial-cell + nudge-from-rest + drag spin-up + clean IC) —
  the plain-z* nudge+drag run reached day 90, so gradual + partial-cell may be the stable
  combination. Partial-cell is kept (real, NEMO-faithful, removes the worst mode).
- **iter 2 (per-term FORCED + partial-cell, job 8106978 — AMPLIFIER = vertadv):** 40-step
  per-term with CORE-II forcing + partial-cell. The Indonesian super-exponential seed is GONE
  (4.4°N/123.5°E KE_PGF_v decays 6.6e-3→3.5e-3). Growth is now ~10× slower (max|u| 68 m/s @
  step 40 vs NaN by step 5 on plain z*). **The consistent AMPLIFIER is vertical momentum
  advection `vertadv`**: once |u|~10-20 m/s it becomes the top term and grows
  (1.5e-3→2.2e-2 over steps 15-40) at the S-Atlantic slope (−36.3/−50.5) + W-Pacific
  (14/136). Residual persistent spurious PGF seeds feed it: equatorial Indian (lat0.3/lon72.5/
  k13, steady 3.46e-3) + slopes. ⇒ Two fix axes: (A) **tame the amplifier** — implicit or
  CFL-limited flux-form vertical momentum advection (`_flux_form_vertical_momentum_advection`,
  ocean_pe_latlon_cgrid.py ~1377) so an overshoot can't run away; (B) tame the seed via
  gradual spin-up. Testing (B) first (no code change): partial-cell + nudge-from-rest + drag
  (job below); the deeper (A) is next if (B) is insufficient/unfaithful.
