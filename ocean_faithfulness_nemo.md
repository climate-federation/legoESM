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
| tripole/eORCA1 | partial-cell coord + adaptive-implicit vertadv (iter3) — amplifier fixed, validating (job 8115940) | pending |
| latlon_bathy | same dynamics path; same fix applies (shared LatLonCGridOceanModel) | pending |
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
1. **BALANCED COLD-START INIT (the indicated fix, iter6 conclusive).** Implement a runner IC
   option `--balanced-init`: geostrophic/thermal-wind velocity from the WOA p′ field
   (u_g = −(1/ρ_0 f)∂p′/∂y at u-pts, v_g = +(1/ρ_0 f)∂p′/∂x at v-pts), equator-tapered
   (regularise 1/f → f/(f²+f_ε²) or zero |lat|<~3-5°); optionally balanced SSH (η from the
   depth-integrated PGF, `eady_uniform` pattern). Reuse `iterate_eos_and_pressure_anomaly` (p′),
   `gradient_x/y_cgrid`, `grid.f_T`/`coriolis_cgrid`. NO dycore change. Test: WOA cold-start
   stays finite + |u| physical (vs rest-IC NaN by day 5). codex-review.
2. If balanced init stabilises → multi-year free run → `compare_omip_nemo.py` SST/SSS vs the
   NEMO 5-yr ref → first real faithful number; then ACC/AMOC/MOC/MLD transports + runoff ungate.
3. If a residual equatorial imbalance remains (f→0 taper region) → digital-filter / incremental
   init, or a short strongly-damped pre-spin to settle the equatorial adjustment.
4. (Done/kept) adaptive-implicit vertadv fix — validated + codex-SHIP; a real robustness
   improvement, NOT the cold-start blocker. Optional follow-up: extend to tracers; fold into
   `_apply_implicit_vertical_mixing` (NEMO trazdf/dynzdf one-solve style).
5. All-grid audit (cubed_sphere/mpas/spectral) — fold in (workflow wnz1vbd2y findings recorded).
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
- **iter 2 (B = drag spin-up CONFIRMED a band-aid):** pcellgrad_8107306 (partial-cell + smc03 +
  nudge60 + Rayleigh drag τ=1d/120d) stayed bounded (max|v|~6-7 m/s) through day 120 then blew
  to 30 m/s the moment drag RELEASED at day 125; weaker drag (τ=5d) NaN'd by day 5. ⇒ drag
  SUPPRESSES the amplifier but doesn't cure it. Pivot to axis (A) — the real, NEMO-faithful fix.
- **iter 3 (FIX (A) IMPLEMENTED — adaptive-implicit vertical momentum advection):** Identified
  the amplifier as the explicit 1st-order-upwind `flux_form_vertical_momentum_advection`
  (vertical.py:740) — NO Courant limit, while the *tracer* TVD path IS Courant-clamped. NEMO's
  exact remedy is **Shchepetkin (2015) adaptive-implicit vertical advection (`ln_zad_Aimp`,
  ON in eORCA OMIP production)**: split w = w_exp + w_imp by a Courant ramp (Cu_min=0.15,
  Cu_max=0.30); w_exp through the explicit scheme (Courant-capped), w_imp through a backward-
  Euler 1st-order-upwind tridiagonal solve (M-matrix → unconditionally stable, monotone,
  conservative). **Implemented** (codex-clean pending): new `shchepetkin_implicit_fraction`,
  `implicit_vertical_advection_ocean` (reuses `thomas_solve`), `adaptive_implicit_vertical_
  momentum_advection` in `ocean/vertical.py`; gated the explicit in-tendency vertadv
  (`ocean_pe_latlon_cgrid.py` ~1357) behind `config.adaptive_implicit_vertadv`; applied the
  operator-split at the step level post-barotropic on the baroclinic perturbation u'=u−U_bar
  using the barotropic-consistent w_baro (`ocean_model_latlon_cgrid.py` ~984); new config flag
  `adaptive_implicit_vertadv` (default False = bit-exact regression); runner
  `--adaptive-implicit-vertadv`; full unit/conservation/stability/AD/flag-off-regression tests
  (`test_adaptive_implicit_vertadv.py`); made `interp_cell_to_vface(f, grid)` fold-aware.
  Unit tests 15/15 + regression 27/27 PASS (job 8115939). codex adversarial review hardened
  the tests (added w=0-exact-identity, machine-precision conservation, rock-leak guards).
- **iter 3 (CONTROLLED EXPERIMENT — vertadv is NOT the cause, RE-DIAGNOSIS):** OMIP ARM A
  (job 8115940, the fix on the exact failing config, NO drag) went NaN by day 5. Per-term
  probe (job 8115992, fix-OFF vs fix-ON, 60 steps, same config) is **decisive**: the max|u|
  trajectories are NEARLY IDENTICAL (step 35: OFF 65 m/s, ON 75 m/s — fix marginally *worse*;
  both ~107-183 m/s by step 60). ⇒ **the adaptive-implicit vertadv fix does NOT change the
  blowup** — so **`vertadv` was a SYMPTOM, not the cause.** It was merely the largest *named*
  tendency term in the prior no-fix diag; it is one of SEVERAL co-equal nonlinear amplifiers
  (KE-gradient feedback −∇(½|u|²), `vertadv`, `vortcor`) that all engage once |u| is O(10).
  With vertadv removed (ARM B), the top term becomes `KE_PGF` at the **S-Atlantic continental
  slope (−36.3°N, −50.5°E)** growing 2.5e-3→1.7e-2 — the runaway continues at the SAME rate via
  the KE-gradient feedback. **ROOT = the spurious PGF SEED over steep topography** (S-Atlantic /
  Indonesian / W-Pacific slopes; ~3.5e-3 m/s² at step 1 with u=0 = pure smc03 partial-cell PGF
  residual). Once it pushes |u| up, ALL the nonlinear terms finish the runaway — removing any
  one (vertadv) cannot help. **The fix must reduce the SEED, not the amplifier.** The vertadv
  fix is KEPT (correct, NEMO-faithful, tested, flag-gated robustness improvement that will
  matter at high res / strong upwelling) but is NOT the OMIP blocker.
- **iter 4 (flat-bottom test = CONFOUNDED; common root is DEEP PGF):** FLAT-BOTTOM WOA
  cold-start (job 8116190) ALSO blows up, FASTER (max|v|=1169 m/s @ 46.9°N by step 10). BUT the
  seed is at k19 (BOTTOM level) over Caspian/North-Sea cells = shallow seas mapped to the
  5500 m flat bottom → WOA stratification extrapolated to depth → spurious deep ρ′ → large deep
  PGF. So flat-bottom is a PATHOLOGICAL test (bad deep IC), NOT a clean topo control. The
  amplifier here is `vortcor` (5.4e-3→0.113 over steps 7-9), NOT vertadv — **reconfirming the
  amplifiers are INTERCHANGEABLE symptoms.** KEY COMMON PATTERN across both realistic-bathy
  (seed at k15-18 over steep slopes) AND flat-bottom (seed at k19 over deep-mapped shelves):
  **the spurious PGF seed always sits at the DEEPEST level** ⇒ root = spurious DEEP baroclinic
  PGF (errors in ρ′ accumulate in p′=∫gρ′dz to the largest value + gradient at depth). Note:
  `iterate_eos_and_pressure_anomaly` HAS an unused `use_depth_dependent_ref` (defaults False →
  constant rho_0 reference; depth-varying ref would shrink ρ′/p′ — mainly a float32-precision
  win, secondary in the fp64 OMIP run). For the REALISTIC case the seed = smc03 partial-cell
  PGF residual on steep slopes (NOT precision; not the flat-bottom IC artifact).
- **iter 6 (DEFINITIVE — PGF is FINE; root = UNBALANCED COLD-START ADJUSTMENT):** Two prior
  axis-B probes came back: (a) bathy smoothing (job 8116430) inconclusive — Laplacian smoothing
  only moved max r-factor 0.995→0.764 (too weak; and total-depth r is the WRONG metric — the
  partial-cell PGF error depends on per-level centroid offsets, not total depth), both doses
  still NaN'd ~day 2. (b) **THE decisive test — horizontally-uniform-stratification REST test on
  the REAL eORCA1 geometry (job 8116497, true baroclinic PGF ≡ 0 ⇒ any velocity is pure
  partial-cell PGF discretisation error): step-1 KE_PGF = 1.0e-6 m/s², max|u| = 4e-4→4e-3 m/s
  over 30 steps, FINITE & STABLE.** Flat-bottom sanity arm = 0.0 exact. ⇒ **the partial-cell
  smc03 PGF on real topography is EXCELLENT (~1e-6 m/s², mm/s — NEMO-class). The PGF is NOT the
  seed.** Therefore the ~3.5e-3 m/s² "seed" in the real WOA cold-start is the **REAL baroclinic
  PGF from WOA's horizontal density fronts**, UNBALANCED because cold-start = rest (u=0) + flat
  SSH (η=0). **ROOT CAUSE (now conclusive): the violent cold-start GEOSTROPHIC ADJUSTMENT from
  an unbalanced rest state** — corroborated by the drag run (damping the adjustment → stable to
  day 120; blew up on drag release). EXONERATED: vertadv/vortcor/KE-grad amplifiers, smc03 PGF,
  partial-cell discretisation, bathymetry steepness. **THE FIX = BALANCED INITIALISATION**
  (thermal-wind geostrophic velocity from the WOA ρ-field, equator-tapered, + balanced SSH) —
  the one prior-session hypothesis NEVER actually tested. Machinery exists: `references.py::
  thermal_wind_shear`, `coriolis_cgrid`, `grid.f_T`, `eady_uniform` balanced-SSH logic; reuse
  `iterate_eos_and_pressure_anomaly` for p′. Implement as a runner IC option (no dycore change).
- **iter 11 (ENERGY BUDGET + B_h + GM — the mode is a fast dissipation-immune ADJUSTMENT mode
  at Brazil-Malvinas; barotropic solver is the last untested structural lever):** Added an
  ONLINE total-KE diagnostic to the per-term probe (committed-ready). Energy budget (job
  8117304): cold-start total KE grows ~LINEARLY (steady ~5e15/step = APE→KE conversion) while
  max|u| grows super-exponentially PINNED at −36.3°N/−50.5°E (Brazil-Malvinas Confluence: sharp
  front + Argentine slope). ⇒ **LOCALIZED mode within a globally-energizing field, NOT a global
  spurious-energy cascade ⇒ EEN/f-split NOT warranted** (avoided the high-risk restructure;
  Hollingsworth A/B already null). Biharmonic momentum viscosity test (job 8117343, tripole has
  B_h=0): B_h=1e13 no effect, B_h=1e14 only ~22% slower (like RK3) ⇒ **the mode is IMMUNE to
  scale-selective dissipation ⇒ NOT a grid-scale 2Δx mode.** GM ruled out by timescale (kappa_GM
  =800 → ~145-day front-flattening vs hours-fast blowup). **DEFINITIVE: the −36.3°N mode is a
  FAST, LOCALIZED, dissipation-immune geostrophic-adjustment / barotropic-gravity-wave mode at a
  sharp front over a slope — immune to EVERY single lever tested (PGF, vertadv, RK3, biharmonic,
  EEN, balanced-init, IC-smooth, GM); only Rayleigh drag masks it (day-120).** The ONE untested
  NEMO structural difference: the **BAROTROPIC SOLVER** — NEMO split-explicit forward-backward
  with AB3-AM4 weights (0.614/0.285/0.088/0.013) that DAMP the fast gravity-wave adjustment;
  legoESM uses implicit-CN (no equivalent fast-mode damping). The cold-start adjustment is
  barotropic-dominated ⇒ this is the next lever. Else: pragmatic persistent-weak-drag run for a
  first caveated NEMO compare (drag proven day-120).
- **iter 10 (RK3 implemented + tested — marginal, NOT the fix; it's a MULTI-COMPONENT gap):**
  Implemented SSP-RK3 (Shu-Osher) outer baroclinic momentum integrator (`momentum_time_integrator
  ='rk3'`, `--momentum-rk3`; config-gated, default euler bit-exact; tests pass; committed-ready).
  Per-term A/B (job 8117140, RK3 vs euler, partial-cell smc03 + forcing + adaptive-vertadv,
  40 steps): **RK3 WORKS (differs from euler) but the trajectory is essentially IDENTICAL** —
  both grow super-exponentially 1.6→90-110 m/s over 40 steps (step40: RK3 86.8 vs euler 111.5,
  only ~22% slower; same top term vertadv@−36.3N/k0). ⇒ **RK3 is NOT the fix; integrator order
  is not the issue.** The cold-start blowup is a **nonlinear robustness gap in legoESM's
  dynamics' response to the unbalanced transient** — every single lever (vertadv, bathy-smooth,
  balanced-init, IC-smooth, RK3) gives the SAME super-exponential trajectory; only the Rayleigh
  drag run reached day 120, and uniform-stratification is stable. NEMO survives the SAME (bigger,
  dt=3600) transient only via its FULL structural stack TOGETHER: split-explicit barotropic
  (AB3-AM4 gravity-wave damping) + EEN energy-AND-enstrophy-conserving vorticity + FCT
  positive-definite tracers + Hollingsworth-KE (nn_dynkeg=1) + RK3 + EVD@100 + adaptive-vertadv.
  legoESM differs structurally: barotropic_solver default `explicit_substep` (OMIP may use
  implicit_cn), AL81 PV-flux vorticity (not EEN), centered KE (not Hollingsworth). **CONCLUSION:
  the faithful cold-start needs MATCHING NEMO'S FULL STACK (esp. the barotropic solver + the
  EEN-vorticity/Hollingsworth-KE energy-enstrophy pairing) — a multi-component dycore effort, NOT
  any single lever.** Pragmatic path to a FIRST NUMBER meanwhile: persistent weak Rayleigh drag
  (the drag run proved day-120 stability) → multi-year → caveated compare vs NEMO 5-yr ref.
- **iter 9 (NEMO COLD-START PROCEDURE decoded — the answer is the TIME INTEGRATOR, IC ruled
  out; workflow w42n63nni, 5 agents over the actual ORCA1 build):** NEMO ORCA1 cold-starts from
  the SAME rest (u=v=0, η=0) + RAW pre-gridded WOCE/Gouretski IC (`woce_*_monthly_init_4p2`,
  native eORCA1, empty weights = no on-the-fly remap) — **NO IC smoothing, NO balancing, NO
  static-stability check, NO interior restoring (`ln_tradmp=.false.`), NO dt ramp, NO Asselin
  (inert under RK3).** ⇒ **my IC-smoothing + balanced-init attempts were the WRONG direction;
  NEMO conditions nothing.** NEMO survives purely via RUNTIME integration robustness:
  **(1) RK3** time-stepping (compile-time `key_RK3` in `cpp_ORCA1.fcm`; `stprk3_stg.F90` 3-stage
  Dt/3,Dt/2,Dt; self-starting, no computational mode) — **legoESM's outer baroclinic step is
  forward-Euler + Matsuno, which has NO stability region for advection/Coriolis/gravity → the
  cold-start adjustment amplifies. THE CORE GAP.** **(2) Enhanced Vertical Diffusion** (EVD,
  `ln_zdfevd`, rn_evd=100 m²/s, tracers-only): N²<0 → K_v=100, collapses static instability from
  step 0. **(3) adaptive-implicit vertadv** (`ln_zad_Aimp`) — ✓ ALREADY implemented (iter3).
  (4) FCT positive-definite tracers (nn_fct_h/v=2); (5) implicit bottom drag (`ln_drgimp`);
  (6) split-explicit barotropic (AB3-AM4 forward-backward, rn_bt_cmax=0.8). dt=3600 single-step.
  Also: vector-form momentum advection + Hollingsworth (nn_dynkeg=1), EEN vorticity, TEOS-10.
  **FAITHFUL FIX PLAN (ranked): (A) implement RK3 (SSP-RK3 / Wicker-Skamarock) for the OUTER
  baroclinic momentum step — the main lift; (B) EVD convective adjustment firing on N²<0 from
  step 0 (legoESM has ocean/physics/convection/enhanced_diffusion.py — verify active in OMIP
  config); (C) DROP the IC crutches — test raw-WOA + rest like NEMO; (D) FCT tracers + implicit
  drag.** A/B test NEMO predicts: RK3+EVD+aimp-vertadv on = stable, off = blowup. Implement A
  (config-gated momentum_time_integrator='rk3') + B, test the raw-WOA cold-start, codex-review.
- **iter 7 (BALANCED INIT implemented — doesn't fix it; narrows to SPURIOUS-SHARP IC):**
  Committed the vertadv fix (2da957ca). Implemented `apply_balanced_init` in the runner
  (`--balanced-init`): level-of-no-motion geostrophic velocity from the WOA p′ (fixed ~1500 m
  reference — NOT per-column seafloor, which gave ±40 m spurious SSH; equator-tapered
  f/(f²+f_ε²); speed-clipped). **Result: ARM A (velocity+SSH) NaN day 2, ARM B (velocity-only)
  NaN day 1 — both FASTER than rest-IC (day 5).** KEY: the geostrophic velocity at the WOA
  fronts is HUGE — 247 m/s at the equator (taper edge), and the clip BINDS globally at 2.5 m/s
  (Southern Ocean + equator) ⇒ the WOA IC mapped onto the tripole (interp + flood-fill +
  partial-cell) has **spurious grid-scale / over-sharp density fronts** implying unphysical
  (>>2 m/s) geostrophic flow. Clipping them to "balance" just injects KE → faster blowup. ⇒
  **refined root: the realistic IC is too ROUGH for the cold start; the dynamics are stable on
  SMOOTH stratification (uniform-strat rest test) but not the rough mapped WOA — balanced or
  not.** NEMO's native-grid Gouretski IC is smoother / NEMO conditions its IC.
- **iter 8 (IC-SMOOTHING test, job 8116796 RUNNING — the indicated fix):** added
  `smooth_woa_ts` + `--woa-smoothing-passes` (horizontal Laplacian on the WOA T,S per level,
  ocean-only). Test: ARM A = WOA + 4 passes + rest start; ARM B = WOA + 12 passes + balanced
  init. PASS = finite + max|u| physical (vs raw-WOA NaN day 5). If smoothing the IC toward the
  (proven-stable) smooth regime holds the cold start ⇒ the fix is IC conditioning (+ optional
  balance); tune the minimal smoothing for faithfulness. If not ⇒ a robust incremental /
  digital-filter init or a damped pre-spin is needed (genuine cold-start-robustness work).
- **iter 5 (axis B — BATHYMETRY SMOOTHING test, job 8116430 RUNNING):** NEMO/ROMS smooth their
  bathymetry to cut the slope (r-factor); the core2 runner used raw eORCA1 e3t_0 (no smoothing).
  Added `make_partial_cell(..., smoothing_passes)` + `--bathy-smoothing-passes` (reuses
  `_laplacian_smooth_2d`; reports max r-factor before/after). Test: WOA cold-start + partial-cell
  + smc03 + vertadv-fix + 8 vs 20 Laplacian passes, ~36 d. PASS = max|u| physical + finite ≥30 d
  (vs un-smoothed control NaN by day 5). If smoothing stabilises ⇒ seed = steep-slope PGF
  confirmed + a (geometry-cost) stabiliser in hand; then tune the minimal smoothing for a
  faithful run. If not ⇒ the deep-PGF/density-anomaly accuracy itself needs work (higher-order
  cubic-spline PGF; depth-dependent reference profile; EOS-at-depth audit).
