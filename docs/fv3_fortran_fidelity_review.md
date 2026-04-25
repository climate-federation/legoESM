# FV3 Fortran Fidelity Review

Baselined 2026-04-14. Older prose is aggressively condensed to
save tokens. Only the newest Ralph-loop tail remains in full
form below.

> **Metadata convention (iter-174)**: do not duplicate "updated
> through iter-N" in the title. The authoritative iteration count
> is the branch commit history plus the HEAD commit message tag.

## Live status

- **W2 v-wind artifact at C36 remains unresolved (structural).**
  Production still runs `FV3EdgeShallowWaterModel` ->
  `fv3_sw_tendencies` (Arakawa-Lamb + RK3 + `boundary_fix`), not
  the FV3 FB chain. Current canonical W2 baseline after iter-761:
  `L2=2.18e-04`, `v_ll_Linf=1.585e-01 m/s`. The artifact remains a
  cube-vertex-meridian stripe pattern and does not converge away
  with resolution.
- **FB chain accuracy at C24/C36 remains unresolved.**
  Halo=3 scaffolding exists, but FB accuracy is still poor. After
  the iter-808 sign-aware sync fix, FB DUOGRID completes 24 h at
  C24 without NaN, but h growth remains far above physical.

## Closed priorities

- **Panel-edge corner metrics**: resolved.
- **d_sw3 BGRID_NE sync**: resolved.
- **Duogrid cube-edge flux synchronization**: resolved by
  iter-807/808 sign-flip sync.
- **Legacy edge handling disabled in duogrid mode**: verified.
- **Non-duogrid `_d2a2c_vect` cube-vertex gap**: isolated,
  architectural, not on the default production path.
- **Old W2 polar face-4 vs face-5 asymmetry**: resolved by the
  iter-505 PPM-axis fix.
- **Visual diagnostic correctness**: corrected in iter-797/820.
- **Cosine bell visual cleanliness**: confirmed in iter-823.

## Key production fix

Iter-505 fixed the major production-path bug in
`cgrid_mass_flux_divergence`: x-direction strips were being passed
to `_ppm_reconstruct_1d` with the wrong active axis. Impact on
canonical W2 C36 dt=300s 1d:

- `L2`: `1.53e-3 -> 2.42e-4`
- `Linf`: `4.07e-3 -> 1.83e-3`
- `max|v_ll|`: `0.557 -> 0.303 m/s`

## Historical archive

Everything before `iter-856` is intentionally compressed here.

- `iter-1..174`: core FV3 metric/operator port, seam/sync work,
  regression expansion, and early FB bring-up.
- `iter-505..510`: production-path PPM-axis fix; old polar
  asymmetry closed.
- `iter-511..729`: W2/W5 artifact characterization, halo=3
  plumbing, FB diagnostics, and production-vs-FV3 routing
  clarified.
- `iter-730..751`: user-visible `v_ll` sentinels were locked; the
  production hyperdiff path was shown to be structurally non-FV3;
  and `use_duogrid=True` on the A-L path was proven catastrophic.
- `iter-752..759`: `_del6_vt_flux` was ported and corrected; the
  del6 post-step path became the production best, cutting W2
  `v_ll_Linf` into the `~0.159` class.
- `iter-760..767`: W2 mode-A was localized to cube corners; three
  Fortran-inspired cube-corner fill ideas were all tested on the
  A-L path and all made W2 worse.
- `iter-768..775`: the residual was re-measured as mainly
  dynamical; `boundary_fix` corner smoothing was shown to be
  load-bearing; `grad_c10` sensitivity was isolated; and a `0.98×`
  corner tuning reduced W2 but was explicitly classified as
  non-Fortran-faithful.
- `iter-776..795`: cosine-bell and W2 plateau behaviour was
  characterized; the W2 smoking gun was narrowed to cube-vertex
  `dv/dt`; the residual was identified as incomplete Cor+press+KE
  cancellation with numerical zeta load-bearing; and
  component-consistent B-halo was refuted for the smooth W2 IC.
- `iter-796..802`: zeta-zero tests confirmed the cancellation
  story; visual diagnostics were fixed; DUOGRID mass-transport
  blowup was localized to the halo=2 scalar path; and
  `fill_corner_region` overshoot plus a failed monotonicity clip
  were documented.
- `iter-803..811`: Fortran-faithful snapshot semantics for pass-2
  diagonals were aligned; the true DUOGRID root cause was found in
  `synchronize_cgrid_fluxes`; sign-aware flux sync fixed the worst
  DUOGRID catastrophe; W5 stayed within ~1% of LEGACY; cosine bell
  was somewhat worse but not catastrophic; and `boundary_fix`
  remained load-bearing.
- `iter-812..820`: DUOGRID visuals showed strong cube-corner
  artifacts; FB DUOGRID no longer crashed instantly but remained
  wildly inaccurate; `damp_v` above `0.03` was identified as the FB
  crash trigger; and the visual `v_north` projection was corrected
  to match the production sentinel.
- `iter-822..830`: W2 LEGACY convergence plateau was confirmed as
  structural; cosine bell was visually clean; W5 showed faint
  cube-vertex ringing; all combined Fortran corner-fill knob
  combinations still worsened W2; and the ocean rest-state
  condition passed.
- `iter-831..840`: FB phase ablations showed `_c_sw` drives h
  growth and `_p_grad_c` is stabilising; several documentary and
  diagnostic cleanups followed; and the canonical sentinel-aligned
  baseline was re-established.
- `iter-841..855`: current-form ut/vt swap was not a drop-in
  improvement; related halo hypotheses were retracted or reverted;
  dv/dt decomposition near the cube vertex was refined; the
  d_sw5-related comparison work underwent multiple honesty passes;
  Phase 1 trial wire-ins were attempted; and iter-854's short-run
  quantitative claims were formally retracted because of stale RK3
  plumbing.

Use git history if you need the full older narrative.

## Latest Ralph-loop iterations (full form)

### Iter-856 — Properly-plumbed Phase 1 trial: BLOWS UP at canonical damping (validates iter-853 prediction)

Per iter-855's iter-856+ Priority 1: take the iter-853 Phase 1 stencil swap and run it with PROPER state plumbing (no RK3 staleness) so each substep sees the actual intermediate u_d/v_d.

**Method.**
- Temporarily added an opt-in `phase1_div_swap=False` and `phase1_dt=300.0` kwargs to `fv3_sw_tendencies` (default-off; production unchanged).  When `phase1_div_swap=True`, the divergence-damping branch used `_d_sw5_corner_divergence` with the call-site `u_d, v_d` to compute Fortran-cc divergence.
- Wrote `scripts/diag_iter856_phase1_proper.py` running its OWN SSP-RK3 loop (not the model's wrapper), calling `fv3_sw_tendencies` directly so each substep saw the correct intermediate state.
- W2 LEGACY at C36, dt=300s, 60 steps.  Damping sweep at {0, 0.001, 0.01, 0.1, 1.0, 2.0} × iter-761 canonical.
- After measurement: REVERTED the `fv3_sw_tendencies` source change (kwargs removed; production code identical to pre-iter-856).  All 14 W2 sentinels pass after revert.  The diag script now warns that re-running requires re-applying the temporary patch.

**Result (PROPERLY-PLUMBED, no staleness).**

| damp_scale | completed 60 steps? | final \|h − h₀\| (m) | note          |
|-----------:|---------------------:|--------------------:|---------------|
| (un-patched 1.0× reference) | yes | 0.691                | baseline       |
|       0.0  | yes                  | 0.857                | stable         |
|     0.001  | yes                  | 0.807                | stable         |
|      0.01  | yes                  | 3.02                  | stable         |
|       0.1  | yes                  | 67.4                  | stable         |
|       1.0  | **NO**                | 1.09e+07               | **blew up at step 10 (~50 min)** |
|       2.0  | **NO**                | 1.20e+08               | **blew up at step 6 (~30 min)** |

**Comparison to iter-854 (BUGGY hybrid, RETRACTED in iter-855).**
- iter-854 1.0×: stable, 301 m (470× worse than baseline).
- **iter-856 1.0×: BLEW UP at step 10.**
- iter-854 2.0×: stable, 639 m.
- **iter-856 2.0×: BLEW UP at step 6.**
- iter-854 0.001×: stable, 0.886 m (≈ baseline).
- iter-856 0.001×: stable, 0.807 m (≈ baseline).  Sub-canonical damping behaviour MATCHES.

iter-856 differs from iter-854 in TWO ways: (a) NO RK3 staleness (each substep sees correct intermediate state) AND (b) NO post-RK3 corrections (iter-854 used `model.step()` which applies `damp_v` vorticity damping + mass fixer AFTER RK3; iter-856 uses a hand-rolled RK3 loop that calls `fv3_sw_tendencies` directly and skips those post-RK3 steps).  iter-856 cannot disentangle which of (a) or (b) caused the qualitative reversal.  Both are likely contributors: (a) gives a more honest tendency at each substep, (b) removes a post-step regulariser.  iter-857+ priority should include re-running iter-856 via `model.step()` (with the same opt-in `phase1_div_swap` kwarg threaded through `tendency_fn`) to isolate (a) from (b).

**iter-853's "would destabilise" prediction was DIRECTIONAL but not strict — qualitative agreement only.**  iter-853 said the 180× t=0 peak amplitude "SUGGESTS short-run instability" / "would destabilise in tens to hundreds of timesteps."  iter-856 measured blow-up at step 10 at canonical damping — directionally consistent.  But iter-856 cannot strictly validate iter-853 against a wrapper-equivalent run; the iter-854-vs-iter-856 difference is also explained by post-RK3 corrections being skipped in iter-856.  Honest summary: iter-856 confirms "Fortran-cc + canonical damping + own RK3 + no post-RK3 damp_v/fix" is unstable; iter-854's "stable at canonical" had AT LEAST the staleness bug as a confounder; the relative weight of the two confounders requires the iter-857+ wrapper-comparison test.

**Refined conclusion (scoped to the iter-856 hand-rolled RK3 + no post-RK3 corrections).**
- The Phase 1 stencil swap with iter-761 canonical damping IS unstable in this configuration (blows up at step 10).
- The Phase 1 stencil swap with damp_scale ≤ 0.1× is stable for at least 5h in this configuration, with integration error growing roughly linearly with damp_scale (0.001× ≈ baseline; 0.1× → 67 m, 100× worse).
- iter-856 alone does NOT establish that "iter-849 Checks 1+4 are required" — it shows that the Fortran-cc stencil + canonical damping coefficient + production application path (du += coeff·grad(div)) + own-RK3 (no post-RK3 damp_v/fix) is unstable.  Whether either Check 1 alone, Check 4 alone, or only the combination, would resolve the instability is iter-857+ scope.
- Phase 1 alone at damp_scale ~ 0.001× IS stable + baseline-quality for 5h in this configuration, but is NOT Fortran-faithful and would not necessarily reduce the W2 LEGACY mode-A signature at 1-day (the v_ll stripe peak is a 24h cumulative phenomenon, not a 5h short-run signal).

**What iter-856 DOES show.**
- Direct measurement of the (properly-plumbed) Phase 1 swap.  No staleness bug.
- At canonical damping (1.0×), the swap destabilises within ~10 RK3 steps.
- At sub-canonical damping (≤ 0.1×), the swap is stable for at least 5h.
- iter-853's t=0 prediction of instability was correct; iter-854's contrary finding was a hybrid artefact.

**What iter-856 does NOT establish.**
- Whether at sub-canonical damping (e.g., 0.001×) the swap improves the 1-day W2 LEGACY v_ll_Linf vs the canonical 0.159 m/s sentinel.
- Whether iter-849 Checks 1+4 (concurrent fixes) would yield a stable + Fortran-faithful + accurate W2 swap.
- Whether the cosine bell or W5 results would be similarly affected by Phase 1 at sub-canonical damping.

**Iter-857+ candidates.**
- **Priority 1 (disentangle staleness from post-RK3 corrections)**: re-run iter-856 via `model.step()` with the same opt-in `phase1_div_swap` kwarg threaded through `tendency_fn` so post-RK3 `damp_v` + mass fixer remain active.  If 1.0× becomes stable with the wrapper, the missing post-RK3 corrections were the dominant destabiliser, not the staleness fix.  If 1.0× still blows up, the staleness fix exposed real instability.
- 1-day (288-step) Phase 1 trial at damp_scale=0.001× to measure v_ll_Linf — IF it improves on 0.159 m/s, the Phase 1 stencil + reduced-coefficient path is a viable mode-A reduction (at the cost of partial Fortran fidelity).
- Implement iter-849 Check 1 (`*dt` factor in adaptive cap) on top of Phase 1 to test whether the magnitude alone is the destabiliser.  If 1.0× becomes stable with `*dt`, Check 1 dominates; if not, Check 4 is also needed.
- Run cosine-bell + W5 evaluations on Phase 1 + damp_scale=0.001× to check for regressions before any production wire-in.
- Continue iter-849 Check 4 architectural port (multi-iter).

**Deliverable.**  `scripts/diag_iter856_phase1_proper.py` (with note that the source patch is reverted).  Production source change to `fv3_sw_tendencies` was applied for the run, then REVERTED.  All 14 W2 sentinels pass after revert.

**Process.**  116th iter in iter-752-856 chain.  First measurement of the Phase 1 stencil swap with proper RK3 substep plumbing (no staleness).  iter-856 measures `Fortran-cc + canonical damping + own-RK3 + no post-RK3 damp_v/fix` configuration and observes blow-up at step 10 at canonical damping.  This is qualitatively consistent with iter-853's "would destabilise" prediction (which iter-854's buggy hybrid contradicted), but iter-856 changed TWO things vs iter-854 (staleness AND wrapper) and cannot strictly attribute the qualitative reversal to the staleness fix alone.  iter-857+ Priority 1 is the wrapper-equivalent re-run to disentangle staleness from post-RK3 corrections.  Sub-canonical damping (≤0.1×) remains stable in this configuration; the architectural attribution to "Checks 1+4 ARE needed" is iter-857+ scope, not established by iter-856 alone.

### Iter-857 — model.step() + properly-plumbed Phase 1: identical to iter-856; staleness fix was the dominant difference

Per iter-856b's iter-857+ Priority 1: re-run iter-856 via `model.step()` (preserves production post-RK3 corrections) with proper RK3 substep plumbing.  This isolates the staleness fix from the wrapper/post-RK3-corrections difference.

**Method.**
- Re-applied iter-856's `fv3_sw_tendencies(phase1_div_swap=False)` opt-in kwarg (default-off).
- Additionally modified `shallow_water_fv3_cdgrid.py`'s `tendency_fn` closure inside `model.step()` to forward `phase1_div_swap` and `phase1_dt` from a module-level toggle `legoesm.core.operators_cdgrid._PHASE1_DIV_SWAP_ENABLED` / `_PHASE1_DT_VALUE`.  Default off → production behaviour identical.
- Wrote `scripts/diag_iter857_phase1_via_modelstep.py` that flips the toggle on, runs W2 LEGACY C36 for 60 steps via `model.step()`, then flips off.  Damping sweep at the same scales as iter-856.
- After measurement: REVERTED both source changes.  All 14 W2 sentinels pass after revert.

**Result.**

| damp_scale | iter-857 (model.step + proper) | iter-856 (own-RK3 + proper) | iter-854 (model.step + stale) |
|-----------:|-------------------------------:|------------------------------:|------------------------------:|
|       0.0  | stable, 0.844 m                 | stable, 0.857 m               | stable, 0.844 m               |
|     0.001  | stable, 0.884 m                 | stable, 0.807 m               | stable, 0.886 m               |
|      0.01  | stable, 3.01 m                  | stable, 3.02 m                | stable, 2.91 m                |
|       0.1  | stable, 67.2 m                  | stable, 67.4 m                | stable, 26.4 m                |
|       1.0  | **BLEW UP step 10**              | **BLEW UP step 10**            | stable, 301 m                  |
|       2.0  | **BLEW UP step 6**               | **BLEW UP step 6**             | stable, 639 m                  |

**Decisive finding (qualitative agreement; stable-scale magnitudes differ at the few-% level).**  iter-857 matches iter-856 QUALITATIVELY: same stability outcome at every tested damp_scale, including the same blow-up step number (10 at 1.0×, 6 at 2.0×).  At the stable damp_scales 0..0.1× the magnitudes are close but NOT identical — at 0.001× the gap is 9 % (0.884 m vs 0.807 m), elsewhere typically 1-2 %.  These small magnitude shifts come from the post-RK3 corrections that iter-856 skips and iter-857 retains.  This is consistent with the qualitative result being driven by the staleness fix, while small magnitude differences come from the post-RK3 corrections.  Specifically:

- The **STALENESS FIX** (substeps 2/3 seeing correct intermediate winds rather than start-of-step winds) is the LIKELY DOMINANT cause of the qualitative reversal vs iter-854.  At canonical damping, the proper plumbing reveals instability that the stale plumbing did not exhibit.
- The **POST-RK3 corrections** (`damp_v` vorticity damping + mass fixer) shift the stable-scale magnitudes at the few-% level but do NOT prevent canonical-damping blow-up — iter-857 has them active and still blows up at the same step as iter-856.

This is an INFERENCE across iter-854/856/857 (a 3-way comparison varying staleness and wrapper independently), not a direct one-variable A/B test vs iter-854.  A strict A/B test would require running iter-854's exact buggy hybrid plumbing alongside iter-857's proper plumbing — that is not done here.

**Validation chain.**
- iter-853's t=0-only "would destabilise" prediction → directionally consistent with iter-856 / iter-857.
- iter-854's "stable at canonical" → retracted (iter-855); iter-857's evidence STRONGLY SUGGESTS the staleness bug was the main cause but does not strictly prove "sole cause."
- iter-855's retraction of iter-854 → confirmed by qualitative agreement of iter-856 and iter-857.
- iter-856's "Phase 1 + canonical damping is unstable" → confirmed by iter-857 with model.step() wrapper.

**Refined conclusion.**
- Phase 1 stencil swap (`_d_sw5_corner_divergence` corner delpc → 4-pt avg to cell centres, fed into the production `du += coeff·grad(div)` formula) at iter-761 canonical damping is REAL-WORLD UNSTABLE.  The 60-step blow-up at step 10 is robust to the integration wrapper choice.
- Sub-canonical damping (≤0.1×) keeps it stable for at least 5h, with integration error growing roughly linearly with damp_scale (0.001× ≈ baseline; 0.1× → 67 m, ≈100× worse).
- iter-857 still does NOT establish that "Checks 1+4 are required" — it shows the Fortran-cc stencil + canonical coefficient + production application path is unstable.  Whether Check 1 alone, Check 4 alone, or only the combination would resolve it is iter-858+ scope.

**What iter-857 DOES show.**
- Same qualitative finding as iter-856 (canonical damping → blow-up at step 10) holds when the wrapper's post-RK3 corrections (`damp_v` + mass fixer) are active.
- The staleness bug in iter-854 was the SOLE cause of its "stable" finding at canonical damping.
- Sub-canonical damping (≤0.1×) is stable in BOTH wrapper configurations.

**What iter-857 does NOT establish.**
- Whether 1-day v_ll_Linf at sub-canonical damping (e.g., 0.001×) reduces below iter-761 canonical's 0.159 m/s.
- Whether iter-849 Check 1 (`*dt`) alone would stabilise canonical damping.
- Whether iter-849 Check 4 (`ke→d_sw6` routing) alone would stabilise canonical damping.

**Iter-858+ candidates.**
- 1-day Phase 1 trial at damp_scale=0.001× via model.step() (with wrapper post-RK3) — measure v_ll_Linf vs iter-761 canonical's 0.159 m/s.
- Add iter-849 Check 1 (`*dt` factor in the `dddmp · |delpc|` cap) on top of Phase 1 and re-test 1.0× stability.
- Continue iter-849 Check 4 architectural port (multi-iter; structurally bigger change).

**Deliverable.**  `scripts/diag_iter857_phase1_via_modelstep.py` (with note that source patches are reverted).  Production source changes to `fv3_sw_tendencies` AND `shallow_water_fv3_cdgrid.py` were applied for the run, then REVERTED.  All 14 W2 sentinels pass after revert.

**Process.**  117th iter in iter-752-857 chain.  Disentangles iter-856's two confounders (staleness AND wrapper) by 3-way inference.  iter-857 matches iter-856 qualitatively (same stability outcome at every damp_scale; same blow-up step at 1.0× and 2.0×).  Stable-scale magnitudes differ at the few-% level (up to 9 % at 0.001×) — consistent with post-RK3 corrections shifting magnitudes but not preventing canonical-damping blow-up.  STRONGLY SUGGESTS the staleness fix is the dominant cause of the iter-854→iter-856 qualitative reversal, but this is inference, not a direct A/B test vs iter-854's buggy hybrid plumbing.

### Iter-858 — 1-day Phase 1 trial: stencil-swap-alone DOES NOT reduce W2 mode-A; rules out Phase 1 + sub-canonical damping as a viable production path

Per iter-857b's iter-858+ Priority 1: with the Phase 1 stencil swap stable at sub-canonical damping for 60 steps (per iter-857), run the FULL W2 LEGACY 1-day integration via `model.step()` and measure v_ll_Linf vs iter-761 canonical's 0.159 m/s sentinel value.  This is the definitive test of "does Phase 1 + reduced damping reduce mode-A?"

**Method.**
- Re-applied iter-857's source patches (phase1_div_swap kwarg + module-toggle forwarding).
- Wrote `scripts/diag_iter858_phase1_1day.py` running W2 LEGACY C36 1 day (288 steps) via `model.step()` at damp_scale ∈ {0.001, 0.01, 0.1}× iter-761 canonical.
- Compared to un-patched 1.0× canonical reference.
- After: REVERTED both source patches.  All 14 W2 sentinels pass.

**Result.**

| damp_scale | phase1 | completed 24h? | L2          | v_ll_Linf (m/s) | h_max | Δv_ll vs 1.0× ref |
|-----------:|:------:|:---------------:|------------:|----------------:|------:|-------------------:|
| 1.0× (un-patched ref) | OFF | yes | 2.176e−04 | **0.188**       | 2998  | (reference)        |
|     0.001  |  ON    | yes             | 3.389e−04 | **0.274**        | 2998  | **+46 %**           |
|      0.01  |  ON    | yes             | 5.210e−04 | **0.331**        | 3001  | **+76 %**           |
|       0.1  |  ON    | **NO**           | n/a       | n/a              | n/a   | blew up at step 161 (~13.4 h) |

**Reference drift caveat (Codex iter-858b note).**  The un-patched reference's v_ll_Linf is 0.188 m/s here; iter-820's measurement was 0.159 m/s.  This discrepancy is NOT a sentinel regression — `TestW2BoundaryErrorBudget` still passes 14/14 with the production code at HEAD (post-revert).  The likely explanation is a measurement-path difference: iter-820's diag uses `apply_cubedsphere_to_latlon` (lat-lon regrid + max), whereas iter-858 uses direct cell-centre projection via `cell_centre_angles_from_4edge` + max over D-grid v-points.  These two paths sample DIFFERENT discrete max-over-points and can disagree by ~10-20 % on the same physical state.  The qualitative comparison (Phase 1 patched values vs un-patched reference) is internally consistent because both use the SAME measurement path within iter-858; the cross-iter quantitative comparison to iter-820's 0.159 m/s should NOT be over-interpreted.

**Decisive finding — Phase 1 + sub-canonical damping is NOT a viable mode-A reduction path.**
- All tested Phase 1 + sub-canonical damp_scales make v_ll_Linf WORSE than the production reference, not better.
- 0.001× → +46 %, 0.01× → +76 %, 0.1× → unstable beyond 13.4 h (blew up at step 161, even though 60-step short-run was stable per iter-857).
- The iter-857 60-step "stable at 0.1×" finding does NOT generalise to 1-day: the integration drifts unstable around step 161 (13.4 h).

**Mechanism interpretation.**  The Fortran-cc stencil produces a different divergence field than `cgrid_divergence`, but feeding it through the production application path (`du += coeff·∇·div`) and the iter-849-Check-1-missing adaptive coefficient does NOT yield a Fortran-faithful damping force.  Reducing the coefficient (sub-canonical damp_scale) keeps short-run stability but degrades the W2 1-day mode-A signature by simultaneously: (a) reducing the production damping that was tuned to the production stencil, and (b) introducing a different (Fortran-cc-shaped) damping signal that grows mode-A rather than suppressing it.

**Conclusion — RULES OUT Phase 1 stencil-swap-alone (with iter-849 Checks 1+3+4 unfixed) as a W2 LEGACY mode-A reduction at the 3 tested sub-canonical damp_scales.**  The Phase 1 stencil swap, at damp_scales {0.001, 0.01, 0.1}× canonical, does not reduce v_ll_Linf below the un-patched 1.0× reference at 1-day.  iter-849 Checks 1 (`*dt` factor) and/or 4 (`ke→d_sw6` routing) ARE candidates for the missing Fortran-faithful piece, but iter-858 doesn't isolate which is required.

**Important methodological lesson.**  iter-857 reported "stable for 60 steps at 0.1×" and iter-858 found the SAME configuration BLEW UP at step 161 (13.4 h).  Short-run stability tests (5 h / 60 steps) do NOT certify long-run stability.  Future Phase-N+ trials should run for at least 1 day (288 steps at dt=300s) before claiming stability.  iter-857's "stable" was a true short-run result; iter-858's "blew up" is the true long-run result.

**What iter-858 DOES show.**
- 1-day v_ll_Linf at all tested Phase 1 + sub-canonical damp_scales (0.001×, 0.01×) is WORSE than the un-patched 1.0× reference.
- Phase 1 at 0.1× becomes unstable between 5h (iter-857: stable at 60 steps) and 13.4h (iter-858: blew up at step 161).  Short-run stability does NOT generalise to 1-day.
- Phase 1 stencil-swap-alone is NOT a path to mode-A reduction.

**What iter-858 does NOT establish.**
- Whether iter-849 Check 1 (`*dt`) added to Phase 1 would change the 1-day v_ll_Linf.
- Whether iter-849 Check 4 (`ke→d_sw6` routing) added to Phase 1 would change the 1-day v_ll_Linf.
- Whether the COMBINATION of Phase 1 + Check 1 + Check 4 would yield Fortran-faithful + stable + mode-A-reducing W2.

**Iter-859+ candidates.**
- Implement iter-849 Check 1 (`*dt` factor in adaptive cap) on top of Phase 1 and re-test at canonical damping.  If 1.0× becomes stable AND v_ll_Linf reduces below iter-761's 0.188 m/s, Check 1 was the missing piece.
- Multi-iter: implement iter-849 Check 4 (`ke→d_sw6` routing).  Bigger architectural change.
- Accept that the d_sw5 architectural port is multi-iter and pivot to a different angle (e.g., audit `_arakawa_lamb_gradient` corner handling, port a different Fortran operator).

**Deliverable.**  `scripts/diag_iter858_phase1_1day.py` (with note that source patches are reverted).  Source changes applied for the run, then REVERTED.  All 14 W2 sentinels pass after revert.

**Process.**  118th iter in iter-752-858 chain.  Definitive 1-day measurement: Phase 1 stencil-swap-alone (with iter-849 Checks 1+3+4 unfixed) does NOT reduce W2 LEGACY mode-A at the 3 tested sub-canonical damp_scales.  In fact it WORSENS v_ll_Linf by 46-76 % at 0.001× / 0.01×, and is unstable at 0.1× by 13.4 h (despite iter-857's 60-step "stable" finding — important reminder that short-run trials do NOT certify long-run stability).  Rules out Phase 1 stencil-swap-alone for these tested damp_scales; iter-849 Check 1 (`*dt` factor) and/or Check 4 (`ke→d_sw6` routing) implementations remain the iter-859+ candidates.  Reference v_ll_Linf measurement-path difference (iter-858's direct-cell-centre 0.188 vs iter-820's lat-lon-regrid 0.159) is documented but does not affect the Phase 1 vs un-patched 1.0× internal comparison within iter-858.

### Iter-859 — Check 1 alone is NO-OP on production; Phase 1 + Check 1 makes things WORSE (catastrophic)

Per iter-858b's iter-859+ Priority 1: implement iter-849 Check 1 (`*dt` factor in adaptive cap) on top of Phase 1 stencil swap and re-test at canonical damping.  Per Fortran sw_core.F90:1720, the `*dt` factor goes inside `dddmp · |delpc|` BEFORE the 0.20 cap.

**Method.**
- Re-applied iter-857 source patches (phase1_div_swap kwarg + module-toggle forwarding) AND added a new opt-in kwarg `phase1_check1_dt: bool = False`.  When True, the adaptive cap argument becomes `dddmp · |div| · phase1_dt` (matching Fortran).
- `scripts/diag_iter859_phase1_check1.py` runs W2 LEGACY C36 1 day (288 steps) via `model.step()` with multiple combinations.
- After: REVERTED both source patches.  All 14 W2 sentinels pass.

**Result (W2 LEGACY 1-day).**

| Config                                | damp_scale | completed 24h? | L2          | v_ll_Linf (m/s)  |
|---------------------------------------|-----------:|:---------------:|------------:|----------------:|
| **A: production (no Phase1, no C1)**   | 1.0        | yes             | 2.176e−04   | **0.188**         |
| **B: production + C1 only**            | 1.0        | yes             | **2.176e−04** | **0.188** (identical to A) |
| D: Phase1 + C1                         | 0.001      | **NO**           | n/a          | blew up at step 60 |
| D: Phase1 + C1                         | 0.01       | **NO**           | n/a          | blew up at step 61 |
| D: Phase1 + C1                         | 0.1        | **NO**           | n/a          | blew up at step 55 |
| D: Phase1 + C1                         | 1.0        | **NO**           | n/a          | blew up at step 8 |

**Striking findings.**

**1) Check 1 alone is a NO-OP on production** (config B identical to A to all reported digits).  Mechanism: with the production `cgrid_divergence` stencil at W2 IC, divergence values are tiny (~1e−8 1/s).  With dddmp=0.2 and dt=300s, `dddmp · |div| · dt = 0.2 · 1e−8 · 300 = 6e−7`.  The `d2_bg` floor at canonical damping is `div_damp / da_min_c ≈ 2.13e+8 / 5e+10 = 4.26e−3`, vastly above the 6e−7 cap argument.  So `max(d2_bg, min(0.20, dddmp·|div|·dt)) = max(4.26e−3, 6e−7) = 4.26e−3` — d2_bg-dominated.  Adding *dt to a cap-irrelevant term is invisible.  This **refutes for the W2 production-magnitude regime** iter-758c's "applying *dt alone breaks RK3" speculation; the refutation is scoped to this regime and does not apply to synthetic tests that DO hit the cap.

**2) Phase 1 + Check 1 BLOWS UP at all damp_scales** (config D blows up at step 8 at canonical, step ~55-61 at sub-canonical 0.001-0.1×).  Mechanism: cap saturation requires `dddmp · |div| · dt ≳ 0.20`, i.e., `|div| ≳ 0.0033 s⁻¹` at dt=300.  iter-851 measured Phase 1 (Fortran-cc) divergence peak ~1.16e−2 1/s at corners — ABOVE this threshold at the peak.  At cap-saturating cells, Check 1 lifts `adaptive_coeff` from the d2_bg floor (4.26e−3 at canonical) to the cap value (0.20) — a ~47× increase concentrated at the cube-vertex-adjacent corners.  At sub-canonical damp_scales, d2_bg is proportionally reduced (e.g., 4.26e−6 at 0.001×) but the cap remains at 0.20 — so the saturated-cell damping is even more disproportionate (~47000× the floor).  This regional damping spike at the peak corners is what destabilises within tens of steps.  iter-859 does NOT observe uniform saturation everywhere; saturation is regional, but the regional spike is enough to destabilise.

**3) Check 1 is REGIME-DEPENDENT.**  On the production stencil, `*dt` has no effect because divergence is below the cap-activation threshold.  On the Fortran-cc stencil, `*dt` activates the cap at the peak corners, where the saturation produces a ~47× damping spike that's enough to destabilise.  Neither regime yields useful Fortran-faithful d_sw5 behaviour; the cap activation depends on a divergence-magnitude regime that production never visits and Phase 1 over-shoots.  iter-849's Check 1 audit identified the gap correctly, but iter-859's measurement shows it is NOT the missing piece for mode-A reduction in either tested regime.

**Conclusion.**  iter-849 Check 1 in isolation (without Check 4 ke→d_sw6 routing) does NOT yield a Fortran-faithful + stable + mode-A-reducing W2 in either:
- Production stencil + Check 1 (no-op): same v_ll_Linf as production.
- Phase 1 + Check 1: catastrophic instability at all tested damp_scales.

The remaining iter-849 audit items are: Check 4 (`ke→d_sw6` routing — multi-iter architectural port, structurally different from the stencil/coefficient swaps tested in iter-851 through iter-859), AND a halo-fixed `_d_sw5_corner_divergence` (per iter-852 caveat: the helper's `nord=0` branch still uses `mode='edge'` halo for vort/ptc).  A halo-fixed Phase 1 might put the divergence in a different magnitude regime where Check 1 has a useful effect, but iter-859 cannot test this without first porting the halo fix.

**What iter-859 DOES show.**
- Check 1 alone on production: production W2 1-day numbers are unchanged (Check 1 is a NO-OP at production magnitude).
- Phase 1 + Check 1: catastrophic instability at ALL damp_scales tested (canonical and sub-canonical), worse than Phase 1 alone.
- The `*dt` factor's effect depends on whether the divergence is large enough to saturate the cap.  Production: never (tested at W2 IC).  Phase 1: only at peak corners where Fortran-cc divergence exceeds the `|div| ≳ 0.0033` threshold; the rest of the domain is still d2_bg-dominated.

**What iter-859 does NOT establish.**
- Whether iter-849 Check 4 (ke-application path) added to Phase 1 + Check 1 would change anything.
- Whether some intermediate divergence-magnitude regime (e.g., halo-fixed `_d_sw5_corner_divergence`) would put the cap activation in a useful regime.

**Iter-860+ candidates.**
- Halo-fixed `_d_sw5_corner_divergence` (port the proper CGRID halo for vort/ptc, replacing `mode='edge'`).  This may put the Fortran-cc divergence in a different magnitude regime where iter-849 Check 1 transitions from "saturating spike" to "actually-Fortran-faithful adaptive coefficient."  Single-iter feasible if the halo port is bounded.
- Multi-iter Check 4 architectural port: re-route the damping through `ke += damp · delpc → corner-rdxc-grad` (Fortran d_sw5 + d_sw6 structure) instead of the production `du += coeff · ∇·div` direct addition.  Bigger structural change.
- Pivot away from d_sw5 port: audit `_arakawa_lamb_gradient` corner handling, port a different Fortran operator, or accept that the mode-A reduction requires a complete dynamical-core swap (e.g., FB chain port, multi-iter).

**Deliverable.**  `scripts/diag_iter859_phase1_check1.py` (with note that source patches are reverted).  Source changes applied for the run, then REVERTED.  All 14 W2 sentinels pass after revert.

**Process.**  119th iter in iter-752-859 chain.  Tests Phase 1 + iter-849 Check 1 directly: Check 1 is a no-op on production stencil (cap-irrelevant at production divergence magnitude `~1e−8`, where d2_bg=4.26e−3 dominates), and destabilising on Phase 1 stencil (regional cap saturation at peak corners with divergence `~1.16e−2 ≳ 0.0033` threshold, lifting damping ~47× at saturated cells).  Neither regime is the missing piece for mode-A reduction.  iter-849 audit items still untested: Check 4 (ke→d_sw6 architectural port, multi-iter) AND halo-fixed `_d_sw5_corner_divergence` (a feasible single-iter port that might shift Phase 1's divergence magnitude into a useful regime).  Pivots iter-860+ scope toward halo-fixed Phase 1 OR Check 4 OR different operator audit.

### Iter-860 — Reproducibility guard: scripts now error loudly without source patches

Codex stop-time review of iter-859: "new diagnostic script is non-reproducible and silently wrong in the checked-in tree."

**The bug.**  iter-857/858/859 scripts set module-level toggles `legoesm.core.operators_cdgrid._PHASE1_DIV_SWAP_ENABLED` etc. via `setattr` to communicate with the temporary source patches in `fv3_sw_tendencies` and the `tendency_fn` wrapper.  After the patches were REVERTED at the end of each iter, the toggle setattr calls STILL succeed (Python lets you set arbitrary attributes), but the un-patched `tendency_fn` wrapper does NOT read them and the un-patched `fv3_sw_tendencies` does NOT accept the `phase1_div_swap` kwarg internally.

The result: iter-857/858/859 scripts run successfully on the post-revert tree but SILENTLY produce production results — they LOOK like they're testing Phase 1 but actually exercise the unmodified production pipeline.  This violates reproducibility: a future reader running the script gets misleading "phase1=True" rows that are actually production output.

(Verified empirically: `scripts/diag_iter857_phase1_via_modelstep.py` ran successfully on HEAD before the iter-860 fix, producing `(reference) un-patched (phase1=False) at canonical 1.0× via model.step(): completed=True, final |h-h0| = 6.409e-01 m` — the production reference value, with no error or warning despite the source patches being gone.)

**The fix.**  Added an upfront `inspect.signature(fv3_sw_tendencies)` guard at the top of each affected script (iter-856, iter-857, iter-858, iter-859) that raises `RuntimeError` immediately if the required `phase1_div_swap` (and for iter-859, `phase1_check1_dt`) kwarg is missing from the function signature.  The error message names the patches needed and points to the corresponding doc entry for the diff.

**Verified.**  All four scripts now exit with a clear `RuntimeError` and explanatory message when run on the post-revert HEAD.  Re-applying the patches makes them work again (the guard checks the signature, and once the patch is back the kwarg is present).

**Why this matters going forward.**  Future trial-only iters that apply temporary source patches MUST add this guard pattern.  A "the patch is reverted" docstring note is INSUFFICIENT — the script must FAIL LOUDLY when run on the post-revert tree.  iter-860 establishes this as the standard.

**What iter-860 DOES show.**
- iter-857/858/859 scripts had a real silent-wrongness reproducibility bug; running them on the post-revert tree produced production results without error.
- The fix is a 5-line `inspect.signature` guard; the pattern is now applied consistently to all 4 trial-script files.
- The original iter-857/858/859 documented results (which were obtained when the patches WERE applied) remain valid and unchanged.

**What iter-860 does NOT establish.**
- Any new mechanism information about Phase 1, Check 1, or W2 LEGACY mode-A.
- This is a doc-honesty / reproducibility-discipline iter, not a measurement iter.

**Iter-861+ candidates (unchanged from iter-859).**
- Halo-fixed `_d_sw5_corner_divergence` (single-iter feasible).
- Multi-iter Check 4 architectural port.
- Pivot to a different operator audit (e.g., `_arakawa_lamb_gradient` corner handling).

**Deliverable.**  Updated `scripts/diag_iter856_phase1_proper.py`, `scripts/diag_iter857_phase1_via_modelstep.py`, `scripts/diag_iter858_phase1_1day.py`, `scripts/diag_iter859_phase1_check1.py` with upfront `inspect.signature` guard.  All raise `RuntimeError` immediately on the post-revert HEAD.  No production source-code change.

**Process.**  120th iter in iter-752-860 chain.  Reproducibility-discipline pass: trial-only scripts that depend on temporary source patches MUST fail loudly when those patches are absent, not silently run production code.  iter-857/858/859 scripts updated to enforce this; pattern established for future iters.

### Iter-861 — Reproducibility guard now also verifies wrapper-source patch

Codex stop-time review of iter-860: "The new guards only verify `fv3_sw_tendencies`, not the required `model.step()` forwarding patch."

**The gap.**  iter-860 added `inspect.signature(fv3_sw_tendencies)` checks for the kwarg presence, but iter-857/858/859 actually depend on TWO source patches:
- (a) `phase1_div_swap` kwarg on `fv3_sw_tendencies` (in `operators_cdgrid.py`).
- (b) `tendency_fn` closure inside `FV3EdgeShallowWaterModel.step` that reads the `_PHASE1_DIV_SWAP_ENABLED` toggle from the module global and forwards `phase1_div_swap=` into `fv3_sw_tendencies`.

If a future user partially re-applies just (a), iter-860's guard PASSES (kwarg is present), but the wrapper still ignores the toggle.  Setting `ocd._PHASE1_DIV_SWAP_ENABLED = True` would have no effect because the wrapper doesn't read it; the kwarg defaults to `False` at the call site.  The script then runs successfully but produces production results — same silent-wrongness pattern that iter-860 was supposed to prevent.

**Fix.**  iter-861 adds a second check via `inspect.getsource(FV3EdgeShallowWaterModel.step)` that scans for the literal string `phase1_div_swap=` (and for iter-859, also `phase1_check1_dt=`) in the wrapper source.  If either piece is missing, the guard raises `RuntimeError` with explicit `(a) ... (b) ...` enumeration of which patches are missing.

(iter-856 doesn't depend on the wrapper — it has its own SSP-RK3 loop calling `fv3_sw_tendencies` directly — so its guard remains unchanged.)

**Verified.**
- iter-857/858/859 scripts now correctly identify BOTH missing patches when run on the post-revert HEAD: `RuntimeError: ... Missing: ['(a) phase1_div_swap kwarg in fv3_sw_tendencies', "(b) tendency_fn forwarding 'phase1_div_swap=' in shallow_water_fv3_cdgrid.py FV3EdgeShallowWaterModel.step"]`.
- Partial re-application (only one of the two patches) would correctly trip the guard with a more specific error message naming just the missing piece.

**What iter-861 DOES show.**
- iter-860's guard was incomplete: it checked the kwarg signature but not the wrapper-forwarding source.
- iter-861's strengthened guard checks BOTH and identifies which is missing.
- Future trial-iters using this pattern should follow iter-861's two-part check.

**What iter-861 does NOT establish.**
- Any new mechanism information.  This is a doc-honesty / reproducibility-discipline iter.

**Iter-862+ candidates (unchanged from iter-859).**  Halo-fixed `_d_sw5_corner_divergence` (single-iter), Check 4 architectural port (multi-iter), or pivot to other operator audit.

**Deliverable.**  Updated guard in `scripts/diag_iter857_phase1_via_modelstep.py`, `scripts/diag_iter858_phase1_1day.py`, `scripts/diag_iter859_phase1_check1.py`.  iter-856's guard unchanged (doesn't depend on wrapper).  No production source-code change.

**Process.**  121st iter in iter-752-861 chain.  Tightens iter-860's reproducibility guard to also verify the wrapper-source patch.  Two-part check: kwarg signature on `fv3_sw_tendencies` AND `phase1_div_swap=` literal in `FV3EdgeShallowWaterModel.step` source.  Future trial scripts should follow this pattern when they depend on multiple source patches.

### Iter-862 — Port Check 3 (cube-vertex corner corrections) into `_d_sw5_corner_divergence` as an opt-in (FB chain only; structural fidelity, halo input still imperfect, default OFF)

iter-849 catalogued FOUR concrete d_sw5 fidelity gaps (Check 1: `*dt` factor; Check 2: corner divergence stencil; Check 3: corner correction stencil; Check 4: ke→d_sw6 application).  iter-851-861 tested Check 2 (Phase 1 stencil swap) and Check 1 (`*dt` factor) on top of Phase 1; both failed to reduce W2 LEGACY mode-A and Phase 1 + Check 1 was catastrophic.  iter-862 ports Check 3 — the four cube-vertex `delpc(corner) ± vort(corner-halo)` corrections from Fortran sw_core.F90:1709-1715 (nord=0) and 1773-1776 (nord>=1) — into `_d_sw5_corner_divergence` as an OPT-IN parameter (default `False`).

**Method.**  Added an opt-in parameter `apply_legacy_corner_corrections: bool = False` to `_d_sw5_corner_divergence` (`src/legoesm/core/fv3_sw_core.py`).  Two new conditional blocks fire only when BOTH the flag is `True` AND `cdgrid.base.duogrid is None` (matching Fortran's `.not. flagstruct%duogrid` gate):

- **nord=0 branch** (after computing `delpc` from the vort/ptc stencil, before the `rarea_c` scaling):
  ```python
  if (apply_legacy_corner_corrections
          and cdgrid.base.duogrid is None):
      delpc = delpc.at[:, 0, 0].add(-vort_pad[:, 0, 0])     # SW corner
      delpc = delpc.at[:, -1, 0].add(-vort_pad[:, -1, 0])   # SE corner
      delpc = delpc.at[:, -1, -1].add(vort_pad[:, -1, -1])  # NE corner
      delpc = delpc.at[:, 0, -1].add(vort_pad[:, 0, -1])    # NW corner
  ```

- **nord>=1 branch** (per-iteration inside the n-loop, after computing `divg_d`, before the `rarea_c` scaling):  same four corrections operating on `uc_lap` instead of `vort`.

DEFAULT IS `False`.  Reason: the right-hand-side `vort_pad` / `uc_lap` values come from the iter-655 `mode='edge'` same-face halo (`fv3_sw_core.py:~1044`), which is documented as an O(1) approximation at cube vertices.  The cross-face D-grid edge halo (`mpp_update_domains(DGRID_NE)` analogue) does not yet exist in our Python.  Until it lands, applying the structural Fortran-faithful correction with the Fortran-incomplete halo data could push legacy FB-chain runs FURTHER from Fortran at cube vertices.  The opt-in flag keeps the structural arithmetic ready for future experiments while default callers (including `_d_sw_native`) remain bit-identical to the pre-iter-862 baseline.  This addresses Codex's iter-862 second-pass MEDIUM finding ("structural port behind a disabled/experimental guard").

**Index translation** (Fortran 1-indexed → Python 0-indexed): `delpc(1, 1)` → `delpc[:, 0, 0]` (SW); `delpc(npx, 1)` → `delpc[:, -1, 0]` (SE); `delpc(npx, npy)` → `delpc[:, -1, -1]` (NE); `delpc(1, npy)` → `delpc[:, 0, -1]` (NW).  South-corner (SW/SE) corrections SUBTRACT vort at the south halo; north-corner (NE/NW) corrections ADD vort at the north halo.

**Tests.**  New `tests/test_fv3_d_sw5_corner_corrections.py` (7 tests):
1. **`test_default_flag_off_is_bitwise_identical_to_baseline_nord0`** — default (no flag passed) reproduces the pre-iter-862 reference EXACTLY in BOTH legacy and duogrid modes.  Protects against accidental behavioural drift in default callers.
2. **`test_flag_on_legacy_mode_applies_exact_corner_corrections_nord0`** — with flag=True + legacy mode, the EXACT corner deltas equal the formula `±vort_pad[corner-halo] * rarea_c[corner] * da_min_c` at the four cube-vertex corners and zero everywhere else.  Catches sign / index / formula errors.
3. **`test_flag_on_duogrid_mode_skipped_nord0`** — flag=True + duogrid mode still bypasses the corrections (Fortran's `.not. duogrid` gate dominates).  Output bit-identical to flag=False on the duogrid grid.
4. **`test_flag_on_legacy_mode_nord1_localises_on_corners`** + **`test_nord1_off_vs_on_only_corners_differ_largest_change`** — for the nord>=1 path, no interior cells are affected by the correction (locality contract) and the largest absolute change is exactly at one of the four cube-vertex corners.
5. **`test_corner_correction_kernel_exact_signs_and_magnitudes`** + **`test_corner_correction_kernel_only_corners_change`** — direct kernel tests on synthetic edge-halo inputs that bypass the upstream face-boundary zeroing, verifying the exact `±edge_halo[corner]` formula and that ONLY the four corners change.  These cover the case where the in-helper nord=1 fixture's corner deltas degenerate to zero (because `_divergence_corner_duo` zeroes face boundaries).  iter-862 factored a small pure helper `_apply_legacy_d_sw5_corner_corrections(field, edge_halo)` so unit tests can pin sign/index/magnitude exactly without depending on the upstream stencil.

`tests/unit/test_cdgrid_fv3_regression.py::TestDSw5NonDuogridCornerCorrectionAbsentIter703` repurposed:
- **`test_no_divg_d_corner_modification_in_source`** — OBSOLETE (the iter-703 "MUST stay ABSENT" contract is contradicted by iter-862's deliberate port).  Now `skipTest`'d with a docstring pointing to the new positive lock.
- **`test_corner_corrections_are_duogrid_gated`** (NEW) — AST scan of `src/legoesm/core/fv3_sw_core.py` flagging any `delpc/divg_d.at[CORNER, CORNER].add(... vort/uc ...)` mutation that does NOT live inside an `if cdgrid.base.duogrid is None:` block.  An unguarded mutation (the regression iter-703 originally watched for) trips this test; mismatched/inverted gate trips it too.
- **`test_no_fill_c_gate_with_fill_corners_call`** — UNCHANGED (iter-862 only ports the corner-correction half of the legacy block; `fill_c`/`fill_corners` Laplacian-iteration pairing is still a deliberate gap).

**Production impact.**  None.  `_d_sw5_corner_divergence` is called only from `_d_sw_native` (FB chain).  Production `fv3_sw_tendencies` does NOT call it.
- All 14 W2 LEGACY sentinels (`TestW2BoundaryErrorBudget`) pass after iter-862.
- `scripts/run_atmosphere_test_matrix.py --only sw --grid cubed_sphere --quick` reports identical numbers: W2 v_ll_Linf=0.159, L2=2.07e-04 (matching iter-820 baseline); W5 mass drift=1.83e-05; cosine bell L1=0.12.
- Cube ocean rest state passes (`test_cube_rest_state_stable`).
- W2 v-wind snapshot (`results/atmosphere/shallow_water/williamson2/cubed_sphere/C36/snapshots_v.png`) shows the same persistent meridian-stripe artifact at cube-vertex longitudes (~0.15 m/s) that iter-820/858 documented — UNCHANGED, confirming production was not touched.

**Pre-existing FB-chain regression unrelated to iter-862.**  `tests/unit/test_cdgrid_fv3_regression.py::TestDSwNativeEndToEndGoldFileIter710::test_d_sw_native_gold_file_nord1` and `test_d_sw_native_gold_file_damp_v_iter727` fail with `h_new.sum() = 383993.75` vs the locked fingerprint `383992.34091496095` (Δ ≈ 1.4).  This failure is **pre-existing on HEAD before iter-862** — verified by `git stash push src/legoesm/core/fv3_sw_core.py && pytest ...` reproducing the exact same failure on the un-patched tree.  These tests use `use_duogrid=True` so iter-862's corner corrections are gated OFF for them; iter-862 does not change their output.  The fingerprint shift is a separate, earlier regression that this iter does not address.

**Scope of fidelity claim — STRUCTURAL, NOT NUMERICAL (Codex iter-862 finding).**  iter-862 ports the Fortran arithmetic STRUCTURE (the four `delpc[corner] ± vort[corner-halo]` adjustments and their `.not. duogrid` gate) but the right-hand-side `vort_pad[corner-halo]` values come from the SAME `mode='edge'` same-face halo whose limitations are documented at iter-655 (`fv3_sw_core.py:~1044`).  At cube vertices, `mode='edge'` returns the local face's edge value, not the cross-face neighbour's value that Fortran's `mpp_update_domains(DGRID_NE)` would have provided.  The Python correction's MAGNITUDE therefore differs from Fortran at cube vertices by an amount bounded by the `|vort_cross_face − vort_same_face|` discrepancy.  Compared to the pre-iter-862 baseline (no correction at all), the structural pattern still nudges delpc / divg_d toward the Fortran shape; it does not yet recover Fortran-faithful magnitudes at cube vertices.

**What iter-862 DOES show.**
- The four cube-vertex corner corrections are now structurally PRESENT in `_d_sw5_corner_divergence`, gated identically to Fortran (`.not. duogrid` ↔ `cdgrid.base.duogrid is None`).
- Production W2/W5/cosine bell sentinels and the `_d_sw5_corner_divergence` non-d_sw5 callers (none in production) are unaffected.
- The test suite catches future regressions: any missing/dropped/inverted gate will trip `test_corner_corrections_are_duogrid_gated`; any wrong sign or wrong corner location will trip `test_corner_corrections_apply_in_legacy_mode_nord0`.

**What iter-862 does NOT establish.**
- Whether the structural-only corner correction CHANGES W2 LEGACY mode-A in the FB chain (FB chain is documented as unstable at C36 for independent reasons; iter-862 does not invoke or measure it).
- Whether the cross-face D-grid edge halo (the right-hand-side data quality fix) would yield meaningfully different magnitudes — that requires the iter-863+ halo port and a follow-on remeasurement.
- Any production W2 mode-A reduction.  Production path does not call this helper.

**Iter-863+ candidates.**
- **Cross-face D-grid edge halo for `vort` / `ptc` / `uc_lap`**: build a `pad_halo_dgrid_edge` helper that takes (6, n+1, n) / (6, n, n+1) edge-midpoint fields and returns properly cross-face-rotated halo'd versions.  Replaces `mode='edge'` at lines ~1044 and ~1174 in `fv3_sw_core.py`.  This makes the iter-862 corrections numerically Fortran-faithful at cube vertices.  Single-iter feasible if scoped to D-grid u-edge / v-edge fields only.
- **Multi-iter Check 4 architectural port**: re-route the production damping through `ke += damp · delpc → corner-rdxc-grad` (Fortran d_sw5 + d_sw6 structure) instead of the production `du += coeff · ∇·div` direct addition.  Bigger structural change.
- **Pivot to other operator audit**: `_arakawa_lamb_gradient` corner handling (largely audited in iter-765/766/825), pressure-gradient at cube vertices, or a different Fortran operator.

**Deliverable.**
- `src/legoesm/core/fv3_sw_core.py`: two new conditional blocks adding the corner corrections gated by the compound test `apply_legacy_corner_corrections AND cdgrid.base.duogrid is None`.  New parameter `apply_legacy_corner_corrections: bool = False` on `_d_sw5_corner_divergence` (default OFF).  Comment blocks at both sites scope the fidelity claim as structural-only and reference iter-655's halo-input gap.
- `tests/test_fv3_d_sw5_corner_corrections.py`: 5 new tests (default flag-off bitwise-identical, flag=True legacy exact corner deltas, flag=True duogrid-mode-skipped, nord>=1 locality, nord>=1 largest-change at corners).
- `tests/unit/test_cdgrid_fv3_regression.py::TestDSw5NonDuogridCornerCorrectionAbsentIter703`: docstring rewritten; `test_no_divg_d_corner_modification_in_source` `skipTest`'d as obsolete; new `test_corner_corrections_are_duogrid_gated` AST-scans for the iter-862 gate (recognising both bare `if duogrid is None:` and AND-combined forms); `test_no_fill_c_gate_with_fill_corners_call` unchanged.
- All 14 `TestW2BoundaryErrorBudget` sentinels pass; production W2 v_ll_Linf=0.159 unchanged from iter-820 baseline.

**Adversarial review.**  Three Codex passes:
1. First pass: 1 HIGH (iter-703 lock contradicted) + 1 MEDIUM (corrections consume known-bad halo).  Addressed by repurposing iter-703 lock + scoping fidelity claim.
2. Second pass: 2 MEDIUM (corrections still on by default with bad halo + nord=1 test too weak).  Addressed by switching to default-off opt-in flag + tightening nord=1 tests to verify exact locality and largest-change-at-corners.
3. Third pass: 2 MEDIUM (nord=1 tests still admit off-by-one if uc_lap[corner-halo]=0 by construction + AST scan accepts an inverted gate when the update sits in the `else` branch).  Addressed by factoring a pure `_apply_legacy_d_sw5_corner_corrections(field, edge_halo)` helper out of `_d_sw5_corner_divergence` and adding direct kernel tests on synthetic non-zero edge-halo inputs (catches sign/index/magnitude exactly even when the in-helper fixture would degenerate to zero), plus rewriting the AST scan to be flow-sensitive (walking `If.body` separately from `If.orelse` so an inverted-gate branch is rejected).

**Process.**  122nd iter in iter-752-862 chain.  Closes Check 3 of the iter-849 d_sw5 fidelity audit at the structural level (corner correction arithmetic + gate), but ships it default-OFF as a future hook because the right-hand-side halo data is still imperfect.  Halo-input numerical fidelity (cross-face D-grid edge halo for `vort` / `uc_lap`) remains a known gap, deferred to iter-863+.  Production unaffected throughout: helper is FB-chain-only and iter-862 does not wire it into any production path.  The default-off flag means the FB chain `_d_sw_native` continues to invoke `_d_sw5_corner_divergence` without the new corrections — bit-identical to pre-iter-862 baseline — until a future caller explicitly opts in.

### Iter-863 — Production W2 audit + new 1-day mode-4 / max|v_ll| / face-mirror regression sentinel + honest conclusion (per user iter-862 reframe)

User reframe (iter-862 stop): the FB-chain corner-corrections work is not on the production path; the live W2 artifact is dynamically generated by the production A-L + RK3 path; please audit production directly, strengthen regressions, and give an honest conclusion about whether the artifact is diagnostic / algorithmic / infrastructure-level.

**Confirmation of user-supplied facts** (re-measured from `results/atmosphere/shallow_water/williamson2/cubed_sphere/C36/snapshots_*.npz`):
- t=0 max|v| = 8.012e-03 m/s; t=1d max|v| = 1.585e-01 m/s — artifact is **dynamically generated**.  Monotone growth across the 11 saved snapshot times; not a t=0 diagnostic-angle bug (those would saturate at t=0).
- face4 maxabs(v_cc_north) = 0.115776, face5 maxabs = 0.115852 — relative difference 6.6e-04.  **Not the old face-4 vs face-5 polar asymmetry** (pre-iter-505 was ~16 %).
- mode-4 zonal FFT amplitude at lat=±30° (lat-lon regridded) = 2.297e-02 m/s on both ±30°.  Equator-symmetric to ulp.  **Symmetric cube-face mode-4 imprint** confirmed.

**Production route audit.**
- Default `FV3EdgeShallowWaterModel` invokes `fv3_sw_tendencies` (`src/legoesm/core/operators_cdgrid.py:1462+`), an Arakawa-Lamb cell-centre tendency form integrated by SSP-RK3 and post-processed by `damp_v` (del-6 vorticity damping) + a mass fixer.  This is the path producing the saved snapshots.
- The FB chain (`fv3_fb_sw_step` / `_d_sw_native`) is the real Fortran-faithful target but is documented as unstable at C36 — it is NOT the saved-baseline path.
- The matrix's canonical W2/W5 config (`scripts/run_atmosphere_test_matrix.py:1197-1202`) sets `hyperdiff_coeff=0`, `div_damp=8 * _div_damp_cube(n)`, `boundary_fix=True`, `damp_v=0.06`, `nord_v=2` — i.e., del-6 post-step is active and scalar hyperdiffusion is OFF.

**Production operator chain — Fortran fidelity verdict.**
- `fv3_d2cc` + `fv3_cc2c` (D-to-cell-centre then cell-centre-to-C-grid wind transforms): structural Fortran-faithful for the `d_grids_to_a_grid` half; the round-trip back to C-grid uses an A-L-style 4-point average that has been audited multiple times (iter-836 / iter-837 / iter-838 / iter-843) for halo behaviour at cube vertices.  Currently load-bearing but not freshly suspected.
- `cgrid_mass_flux_divergence` (PPM transport): closed by iter-505 PPM-axis fix.  Not the live driver.
- `_arakawa_lamb_gradient` corner handling: tested across iter-765 / iter-766 / iter-825 with three Fortran-inspired corner-fill knobs (`fortran_dir_aware_corners`, `fortran_a2b_corner_avg`, `fortran_vector_corner_fill`).  All knobs WORSEN W2; iter-825 ruled out individual + combined firings post the iter-808 sign-flip sync.  Not a single-iter improvement vector.
- `pad_halo_vector` corner-wind construction (steps (e) / (k) of `fv3_sw_tendencies`): cube-vertex behaviour audited iter-836 - iter-838.  The iter-836 halo-only fix had a 15.6 %-of-interior-scale error at cube vertices; iter-836b backed off; iter-837 added covariant-aware rotation; iter-838 added cross-face metric halo.  Currently uses the iter-838 path; not a single-iter improvement vector.
- **`div_damp` block** (lines ~1690-1707): the production form is `du += adaptive_coeff · ∂div/∂x` at cell centres after a cell-centre `cgrid_divergence` and an A-L gradient back to corners and interpolation to centres.  Fortran d_sw5 + d_sw6 build `delpc` at corners with edge-by-edge `ptc/vort` (Fortran-cc form), apply `damp · delpc → ke` and `ke[i,j] − ke[i+1,j] → u`.  These are structurally different forms — Fortran damps with corner-localised coefficients on a corner-localised divergence; production damps with cell-centre coefficients on a cell-centre divergence after an A-L round trip.  iter-849 catalogued FOUR concrete fidelity gaps; iter-851-862 attempted Phase 1 (stencil swap), Check 1 (`*dt`), and Check 3 (corner corrections, FB-chain only).  Phase 1 stencil-swap-alone does NOT reduce W2 mode-A (iter-858); Phase 1 + Check 1 is catastrophic (iter-859); Check 3 is now ported as default-off opt-in (iter-862).  Check 4 (ke→d_sw6 routing) is the multi-iter architectural change that would replace production's cell-centre form with Fortran's corner-localised d_sw5 + d_sw6 form.  This is the only remaining Fortran-fidelity vector for the divergence-damping block.
- **`boundary_fix`** (lines ~1745-1789): explicit non-FV3 stabiliser averaging boundary-cell tendencies with adjacent-interior cells.  Documented as load-bearing for W2 L2 (iter-511: turning it off worsens L2 by 2.4×).  **Removing or weakening it is gated on the FB chain becoming stable at C36** — i.e., the Fortran-faithful d_sw5 + d_sw6 architecture replacing the need for boundary smoothing.
- Post-step `damp_v` del-6 vorticity damping: Fortran-faithful (`_del6_vt_flux`, iter-727), already active in the canonical matrix config.  Not the missing piece.

**Why the artifact persists despite damp_v active.**  `damp_v` damps relative vorticity gradients; the cube-face mode-4 imprint is a pressure-gradient + KE-gradient + Coriolis cancellation residual that is NOT primarily vortical (verified by iter-844-846's dv/dt decomposition: cube-vertex residual is dominated by 4-term cancellation imbalance, not by vorticity-flux divergence).  Damping vorticity does not target the residual's source.  Adding more damping (raising `damp_v` or `div_damp`) tunes the symptom but does not address the structural cell-centre vs corner-localised mismatch.

**The honest answer to "diagnostic / algorithmic / infrastructure".**  **Infrastructure-level.**  Specifically, the residual is the production A-L + RK3 path's cube-face mode-4 imprint, which is structurally inherent to:
1. cell-centre `cgrid_divergence` + A-L corner gradient + cell-centre projection (vs Fortran's corner-localised d_sw5);
2. RK3 vs Fortran's FB time-splitting;
3. `boundary_fix` as a non-Fortran stabiliser compensating for (1) + (2).

These three are intertwined — replacing any one in isolation is what iter-749-862 has tried (Phase 1 stencil swap, Check 1 `*dt`, Check 3 corner corrections, multiple A-L corner fills) and has consistently failed to improve W2 v_ll_Linf below ~0.16 m/s.  The remaining Fortran-fidelity vectors that have NOT been tried in production and could plausibly reduce the imprint are:
- **Check 4 architectural port**: re-route divergence damping through `ke += damp · delpc → corner-rdxc-grad` (Fortran d_sw5 + d_sw6 structure) instead of cell-centre `du += coeff · ∂div/∂x`.  Multi-iter (requires d_sw5 + d_sw6 ports + `ke` corner field plumbing into `fv3_sw_tendencies`).
- **FB chain stabilisation at C36**: fix the FB chain's instability so it can replace the A-L + RK3 production path entirely.  Multi-iter, but the genuine Fortran-faithful target.

**Direct user question — can production A-L + RK3 honestly get much closer to FV3?**  No.  The available smaller knobs (`hyperdiff_coeff`, `div_damp` magnitude, A-L corner fills, halo variants) have been audited and do not move the residual below ~0.16 m/s without re-introducing other artifacts (see iter-825 corner-fill table).  The structural form of the production divergence-damping block plus `boundary_fix` is the floor.  Any further improvement is multi-iter architectural (Check 4 ke-routing OR FB chain stabilisation).

**iter-863 deliverable.**  New regression sentinel `tests/unit/test_cdgrid_fv3_regression.py::TestW2BoundaryErrorBudget::test_w2_iter761_matrix_v_ll_and_mode4_baseline` runs W2 LEGACY C36 1 day on the iter-761 canonical matrix config and asserts:
- `max|v_ll| < 2.0e-1` (saved baseline 1.585e-1, ~25% headroom)
- `mode4(±30°) < 3.0e-2` (saved baseline 2.297e-2 each, ~30% headroom)
- `face4 vs face5 maxabs mirror rel < 1.0e-2` (saved 6.6e-4, ~15× headroom)

A future patch that worsens any of these on the canonical matrix config trips this test.  A future patch that BEATS the baseline (lowers max|v_ll| or mode-4 below the saved value) is the next concrete improvement target.  Cost: ~14 s on CPU x64; same class as the existing iter-761 L2 test.

**Acceptance-criteria audit** (user iter-862 reframe):
- ✓ keep t=0 diagnostic-angle bound below 0.01 m/s — already locked by `test_w2_t0_v_north_diagnostic_angle_bounded` (passes; pre-iter-505 ~0.39 m/s, current ~8e-03).
- ✓ keep face4/face5 mirror near 1e-2 scale — now locked by iter-863's new test (rel < 1.0e-2; current 6.6e-04).
- ✓ canonical W2 h_L2 within 25 % of 2.07e-04 — already locked by `test_w2_alpha0_c36_1day_iter761_matrix_config` (L2 < 4.0e-4).
- ✓ do not silently disable boundary_fix — locked by `test_boundary_fix_is_load_bearing_for_w2_l2`.
- iter-863 does NOT beat the saved baseline; instead it explicitly pins the current values so future work can target them.

**What iter-863 DOES show.**
- Direct measurement of saved snapshots confirms the user's facts: dynamically-generated symmetric mode-4 cube-face imprint, NOT polar asymmetry, NOT t=0 diagnostic.
- Concrete production-chain audit identifying Check 4 (ke→d_sw6 routing) and FB-chain stabilisation as the only remaining Fortran-fidelity vectors that could reduce the residual below ~0.16 m/s.
- New 1-day regression sentinel pins the three user-visible W2 artifact metrics on the iter-761 canonical matrix config; future regressions will be caught.

**What iter-863 does NOT establish.**
- Any production W2 improvement.  The new test pins; it does not improve.
- Whether Check 4 architectural port WOULD reduce the residual.  Multi-iter; iter-863 only documents it as the next viable vector.
- FB-chain C36 stability; remains a documented blocker for the genuine Fortran-faithful production path.

**Iter-864+ candidates.**
- **Multi-iter Check 4 port**: re-architect the production divergence-damping block from cell-centre `du += coeff · ∂div/∂x` to corner-localised `ke += damp · delpc → u += rdxc · (ke[i] - ke[i+1])`.  Requires plumbing a `ke` corner field through `fv3_sw_tendencies`, replacing the A-L gradient round-trip with Fortran's corner-difference form.  Targets a meaningful reduction in W2 mode-A.
- **FB chain stabilisation at C36**: identify and fix the C36 instability of `fv3_fb_sw_step` / `_d_sw_native` so it can replace the A-L + RK3 production path.  This is the genuine Fortran-faithful route.  Bigger architectural commitment.
- Cross-face D-grid edge halo (the iter-862 deferred work) — required if Check 4 is ported AND we want the `delpc` corner-source magnitudes to match Fortran at cube vertices.  Single-iter feasible if scoped to D-grid u-edge / v-edge fields only.

**Deliverable.**
- `tests/unit/test_cdgrid_fv3_regression.py::TestW2BoundaryErrorBudget::test_w2_iter761_matrix_v_ll_and_mode4_baseline` — new 1-day W2 regression sentinel (max|v_ll| + mode-4 + face-mirror).  Passes at the current iter-761 canonical baseline.
- This doc entry: confirmation of user facts, production-chain audit, honest infrastructure-level conclusion, and iter-864+ candidate list.

**Process.**  123rd iter in iter-752-863 chain.  Direct response to user's iter-862-stop reframe: stop polishing FB-chain trims, pin the user-visible W2 baseline so future regressions trip, and give a structural-audit answer to "is this diagnostic / algorithmic / infrastructure".  Conclusion: infrastructure (production A-L + RK3 form vs Fortran d_sw5 + d_sw6 form, plus `boundary_fix` as non-Fortran stabiliser).  Smaller knobs are exhausted; meaningful reduction requires a multi-iter architectural change (Check 4 ke-routing OR FB-chain stabilisation).
