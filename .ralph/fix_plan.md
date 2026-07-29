# Ralph Fix Plan — DINO/NEMO fidelity sweep to AT BAR 45 | DEBT 0 | UNMEASURED 0

## READ FIRST, EVERY LOOP (these are the spec, this file is only the queue)
- `.claude/ralph_dino_fidelity_to_bar_task.md` — goal, method, established facts, FALSIFIED list,
  guardrails, per-row STOPPING RULE, ESCALATE list.
- `.claude/skills/oracle-fidelity/SKILL.md` — Rules 0, 1, 1b, 1c, 1d, 1e. Each was written after a
  real mistake that cost hours or produced a wrong committed number.

## The gate IS the definition of done — drive off it, not off impressions
```
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/fidelity_bar_gate.py
```
Report its `AT BAR n | DEBT n | UNMEASURED n` line at the START and END of every loop.
DONE = exit 0, `DEBT 0 | UNMEASURED 0`, and ZERO rows under "AT BAR on CANCELLING statistics only".

**NEVER weaken the gate** — not `BAR_CORR`, not `BAR_RATIO_EPS`, not `BAR_PER_ELEM_EPS`, not by
deleting a `PER_ELEMENT` entry, not by flipping a `BINARY_GATES` verdict. Those constants ARE the
task. `--self-test` must keep passing every loop.

## High Priority (in order)
- [ ] **THE ACC LEVER — the wind does not survive the barotropic pathway. HIGHEST VALUE IN THE
      CAMPAIGN.** ACC momentum budget measured 2026-07-29 (acc_momentum_budget.py), both models fed
      the SAME NEMO Y5 state, channel band rows 14-48 (-64.44..-45.35 degN, chosen objectively as
      rows with no land at any longitude).
      Channel- and depth-integrated zonal force difference (lego - NEMO) [m^3/s^2]:
          adv +1.24 | vor -2.40 | ldf -0.002 | hpg +1.09   -> INTERIOR total -0.075, **0.0% share**
          spg -1240 | zdf -13856                            -> SURFACE/BAROTROPIC **100.0% share**
      Per-element the interior terms are near-exact too: adv+hpg err_norm 7.6e-5 (corr 1.000000),
      vor 1.7e-3, ldf 4.5e-2, and the four-term sum vs NEMO's D06 accumulator 1.24e-3.
      MECHANISM: legoESM's wind ENTERS correctly — its explicit du_dt carries +1.3845e4 vs the
      analytic channel input +1.3856e4 (0.9992x) — but the REALIZED channel tendency after the
      barotropic + vertical stages is -1238 vs NEMO's +13859, a shortfall of -1.09x the wind input.
      The wind is input correctly and then does NOT SURVIVE the free-surface/barotropic pathway.
      NEXT: which operation removes it — the F_slow depth-mean split, the substep recurrence, the
      barotropic bottom drag, or the mean re-imposition? Note this is the SAME pathway as the
      already-found missing `zu_trd` subtraction (dynspg_ts.F90:304/:367), which is now a prime
      suspect rather than a loose end.
      CAVEAT: instantaneous budget at Y5 with both models on a shared state. It establishes which
      term differs THERE; it does NOT prove that term caused the 5-year from-rest divergence, and a
      term matching here can still differ in the spin-up regime. Re-run the same budget on a
      from-rest state before treating it as closed.
      THREE TRAPS caught and now gated in the script, each of which would have produced a false
      ranking: (a) NEMO stage 8 is BAROCLINIC-ONLY (dynzdf.F90:150-151 strips uu_b(Kaa) under
      ln_drgimp+ln_dynspg_ts) — unrestored it ranked zdf at 97.7% on a depth-uniform artifact;
      (b) legoESM's vortcor_u ALREADY carries (f+zeta) x u, so adding coriolis_cgrid double-counts
      f (RMS 2.001x at corr 0.99998); (c) the bridge leaves tau_x_prev=None, which SILENTLY DROPS
      THE WIND FROM STEP 1 — seeded per NEMO's before:=now rule.
- [ ] **THE BAROTROPIC SEED — now the single highest-value target (3 rows).**
      MEASURED 2026-07-29 (spg_substep_chain.py): `dyn_spg_ts` is NOT the defect. The velocity
      error is already at the LOOP-ENTRY SEED (err_norm u 2.31e-2) and flat through substep 1
      (2.31e-2) to final (3.09e-2). Substep count matches exactly (23==23, 68==68); alignment
      sharp at (0,0) on both a U-face and a T-point field; ssh EXACT at seed => velocity-specific,
      not shared with the eta/PGF chain. `puu_b`, `un_adv` and `pssh` are ALL INHERITED from the
      seed — the solver faithfully propagates a wrong initial condition.
      **CORRECTED AGAIN 2026-07-29 — it was the LEGOESM_NEMO_E3T DEFAULT.** The probe ran with
      the env var UNSET -> NEMO's analytic e3t_1d, which differs from the e3t_0 NEMO runs on by
      12.9% below k=25. A/B: e3t=off seed 2.3109e-02 / substep-1 2.3131e-02 / final 3.0949e-02;
      **e3t=both seed 2.1872e-16 / substep-1 2.9656e-04 / final 1.9918e-02.** On the REAL ladder
      the seed is EXACT and the error ACCUMULATES through the substeps — the opposite branch.
      dyn_spg_ts DOES have a substep problem; the seed does not.
      FALSIFIED: u-face wet-level count matches exactly (0/9758) — not the bug-#18 mask family.
      NOT create_levy_stretched_z_star — the bridge never calls it (dz_ref == NEMO e3t_1d to
      9e-16). That constructor feeds DINO's STANDALONE path only, where the missing second
      mi96 pass (zgr_lib.F90 re-anchor at rn_hco=1000 m) may still be a separate real defect —
      worth its own row, NOT yet measured.
      GUARDED: precision_gate.require_explicit_e3t_mode() now refuses an inherited default
      (4th contaminated measurement).
      **CHAIN CLOSED 2026-07-29: the SLOW FORCING `zu_frc` owns the worst row.**
      zu_frc err_norm 8.03e-03 -> propagated through ONE substep (rDt_e=117.391304 s) predicts
      substep-1 2.9661e-04 vs measured 2.9656e-04 (ratio 1.000) -> x68 predicts final puu_b
      2.0166e-02 vs measured 1.9918e-02 = **101.2% explained**. The entire 1.3e-2 gap reduces to
      ONE quantity.
      OPEN: the V-component does NOT close the same way (12.6% explained) — `pvv_b` has a second
      cause. Do not assume symmetry with u.
      **STAGE 7 DONE 2026-07-29 — static-weight hypothesis FALSIFIED; a structural gap found.**
      Swapping NEMO's static e3u_0/r1_hu_0 onto legoESM's own du_dt made it 44x WORSE
      (8.0266e-03 -> 3.5264e-01). Static and live must stay PAIRED with their own RHS; the weight
      is not a drop-in. Residual sits at sloped-bathymetry columns near the channel walls, not the
      seam (only 10% touch the wrap columns).
      **NEXT / strongest lead: legoESM has NO `zu_trd` subtraction.** NEMO does
      `zu_frc = zu_frc - zu_trd*ssumask` (dynspg_ts.F90:304, :367); our F_slow_u is finalised at
      ocean_model_latlon_cgrid.py:2841 with nothing removed after. An ABSENCE, not a sign/mask
      bug. BEFORE adding anything: check whether legoESM's formulation makes the removal
      unnecessary by construction (i.e. never adds the component NEMO removes) — a wrong "fix"
      here would be easy and would look plausible.
      HINT ONLY, not evidence: RSS of dyn_ldf/dyn_vor/dyn_adv ZAD/dyn_cor_2d = 6.49e-3 vs zu_frc
      8.03e-3, same order — consistent with zu_frc being their depth-mean, but NO per-level
      correlation was run.
      CONFIRMED ASYMMETRY: zv_frc err_norm 5.4290e-04, ~15x smaller than zu_frc — pvv_b's cause is
      genuinely separate.
      (original question) why is zu_frc off by 8.03e-03? **KEY SOURCE FACT (verified from cpp_DINO.fcm =
      `key_qco key_vco_3d`)**: under key_qco NEMO builds it with the **STATIC** reference ladder --
      `SUM(e3u_0 * puu(Krhs) * umask) * r1_hu_0` (dynspg_ts.F90:336) -- NOT the live e3u(Kmm),
      which is the `# else` branch DINO does not compile. The in-source comment gives the reason
      ("e3. are substitute by 1D arrays and can't be used in SUM operand"). So for THIS term the
      oracle's choice is STATIC, the OPPOSITE of the three live-ladder fixes already made — do not
      "fix" it in the direction that felt right last time. Also checking the
      `zu_frc -= zu_trd*ssumask` removal (:304/:367).
      IF the weighting is not the cause, zu_frc is the depth-mean of the 3-D momentum RHS, and
      several DEBT rows measure pieces of that same RHS (dyn_ldf 3.9e-3, dyn_vor EEN 1.2e-3,
      dyn_adv ZAD 4.9e-3, dyn_cor_2d 1.2e-3) — so the barotropic rows may be a DOWNSTREAM
      CONSEQUENCE of the momentum rows. That would be a big structural unification; it needs the
      arithmetic shown, not asserted.
      (original) why is zu_frc off by 8.03e-03? It is the vertically-integrated slow
      (baroclinic) momentum trend — candidates are the terms summed into it and the integration
      weights. The e3t ladder is now correct, so it is NOT that.
      (growth shape, for the record)
      GROWTH SHAPE MEASURED 2026-07-29: final/substep-1 = 67.2 against 68 substeps => the
      accumulation is **LINEAR**, i.e. a CONSTANT ~2.97e-04 added per substep, NOT amplifying
      feedback or an instability. That is the signature of a term held FIXED across the loop
      being slightly wrong — in NEMO's split-explicit scheme that is the SLOW FORCING
      (zu_frc/zv_frc/ssh_frc, computed once before the loop and applied every substep), which is
      dumped. Testing that first; if it is clean the error is in a per-substep term instead
      (SSH gradient, in-loop Coriolis, bottom drag, the forcing add, or the ssh update).
      (superseded: RESOLVED TO (B): the AVERAGING OPERATOR owns it; the 3-D velocity is
      BIT-IDENTICAL to NEMO (err_norm 0.0 at all 35 levels, both components).** Since the
      numerator is exact, the entire 2.31e-2 lives in the thickness weighting
      (`_depth_average_to_faces`). Not upstream dynamics, not the bridge, not the solver.
      NEXT: per-column `sum_k(h_face)` vs NEMO `sum_k(e3u(Kbb))`, and the u-face WET LEVEL COUNT.
      DINO is FULL-STEP so e3u_0 == e3t_0 and the min-rule should agree level-for-level — which
      points at WHICH LEVELS ARE COUNTED WET at a u-face (bathymetry steps), i.e. the same mask
      family as bug #18 (ULP tie marking the deepest dry level active) and ldf_slp's zcj count.
      DEAD END, do not retry: `stp_dump_07_dynspg_u/ub.bin` are NOT the entry velocity —
      stpmlf.F90:288-294 dumps `uu(Naa)`/`uu_b(Naa)` and :403-406 shows Naa == Nrhs in the step
      body, so `_u.bin` is the momentum RHS accumulator (~1e-6) and `_ub.bin` duplicates
      spg_dump_puu_b_final.bin.
      (historical framing: the seed is `sum_k(u*h_face)/sum_k(h_face)`, two separable causes — (A) the 3-D VELOCITY entering it is already wrong, or (B) the
      WEIGHTING/averaging is wrong. Measuring (A) FIRST against `stp_dump_07_dynspg_u/ub.bin`
      (NEMO's 3-D velocity at the dyn_spg_ts entry); if that is already ~2-3% off, the seed is
      inherited and ALL the barotropic code is exonerated.
      DOWNGRADED CANDIDATE: `barotropic_seed_face_depth` ("min_rule" vs "nemo_ssh_avg") is
      DOCUMENTED AND MEASURED INERT away from the `min_water_column_m` floor — a per-column
      scalar cancels identically in the weighted mean — so it can only bite on thin/shelf
      columns. DINO is a deep basin, so it is a WEAK explanation for a domain-wide 2.3%. Test it
      only if the pattern turns out to be shelf-concentrated. (I had called it the "leading
      candidate" last iteration; reading its own docstring says otherwise.)
- [x] **DONE 2026-07-28/29 — metric_convention pass (human APPROVED: NEMO match).**
      `nemo_isotropic` already sets `dy_T`/`dy_u`/`area_T` and now `dy_v = R*dlon*cos(lat_v)`
      (NEMO `pe2v = pe1v`, `usrdef_hgr.F90:117`). REMAINING: wire the DINO bridge to select it
      (`bridge_nemo_to_legoesm_topo` defaults to `"exact"` regardless of the recipe card — a
      harness gap), then ONE controlled pass recording before/after for EVERY affected row
      (`vslp`, `wslpj`, `zaj`, `ssh_nxt`, `dyn_cor_2d`, anything else that moves).
      DONE: the bridge now defaults to metric_convention="auto" and DETECTS the convention from
      the oracle's own mesh_mask (detect_metric_convention: DINO has e1t == e2t EXACTLY), which
      also stays correct for non-isotropic NEMO configs and cannot drift like a recipe card.
      dy_v matches NEMO's e2v to 0.000e+00. All four ldf_slp rows are PRODUCTION-TRUE and below
      the 1e-9 per-element bar: wslpi 3.427e-10, wslpj 2.770e-10, uslp 2.200e-10, vslp 3.265e-10
      (vslp was 1.541e-06 -- 4700x). Remaining gap on them is the |x|ratio (1.7e-5..4.3e-5),
      which is the documented CONDITIONING tail, not a transcription error.
- [x] **DONE 2026-07-29 — the 6 cancelling-only AT BAR rows** — cheapest real progress. They need only a per-element
      measurement. Five claim "exact"/"bit-exact" in their notes, but a CLAIM IS NOT A MEASUREMENT,
      and that exact gap is what let `bn2` sit falsely AT BAR for weeks.
      RESULT: all six clear the 1e-9 bar; NO cancellation was hiding a defect this time --
      but that is now a MEASUREMENT rather than a claim. The gate's "AT BAR on CANCELLING
      statistics only" line is GONE: all 11 AT BAR rows are per-element proven.
      Two caveats recorded in the gate, both honest limits rather than failures:
      the ATF `ssh` leg is TAUTOLOGICAL (one unknown solved then re-substituted, so it
      certifies transcription only -- T/S do not share this), and `sfx` was reconstructed
      from the production A_S/dino_S_star because no first-class legoESM sfx array exists.
      Two probe-side wet-mask bugs were found and fixed mid-measurement (u-face periodic
      seam, v-face wall row) -- both harness, not legoESM.
- [ ] **Instrument NEMO for the big rows** — but CHECK FIRST, more is already dumped than the
      queue assumed. `dyn_spg_ts` needs NO rebuild: its whole substep chain is already on disk
      (spg_dump_{zu_frc,zv_frc,ssh_frc, un_e_init,vn_e_init,sshn_e_init, ub_substep1,vb_substep1,
      ssh_substep1, puu_b_final,pvv_b_final,pssh_final,un_adv_final,vn_adv_final}.bin), which is
      enough to answer the decisive question (error at substep 1 vs accumulated over ~69).
      Still likely needed for: `zdftke`, `dyn_vor`,
      `traadv_fct`. Add `WRITE` dumps of the INTERMEDIATES following the
      `cfgs/DINO/MY_SRC/ldfslp.F90` pattern (units 8840-8848, `l_1226_raw_dump_done` first-call
      flag), rebuild, rerun, register each dump in `legoesm.ocean.fidelity.time_levels` with its
      `file:line`. This is the single highest-leverage action: `ldf_slp` closed in hours BECAUSE
      its chain was dumped, while the 3e-6 band resisted for weeks on endpoints alone.
- [x] **DONE 2026-07-29 — audited every static-depth-ladder consumer. The prior PAID OFF: a THIRD
      site found.** `eos.compute_ocean_rho`'s `eos_depth="geometric"` branch (eos.py:2336) read the
      static `t_depth_ref`. Reachable ONLY from `fidelity/tendency_probe.py` — i.e. the ORACLE
      TENDENCY COMPARISON — so probes there measured the model against a different density
      convention than the model itself now uses. Fixed, gated identically (static when no
      free-surface info is on the state, `|z_full_ref|` when no fidelity ladder). 219 tests pass.
      The remaining `t_depth_ref`/`z_full_ref` hits are INITIALISATION (`init.py`,
      `init_woa.py`, `init_latlon_cgrid.py`, `bathymetry.py`) — building a T(z) profile at t=0,
      where a static ladder is CORRECT. Audit complete; do not re-run it.
- [x] DONE — `vslp` residual: closed by the metric pass (1.541e-06 -> 3.265e-10).
- [x] **DONE 2026-07-29 — `ldf_eiv aeiu` re-measured. IMPROVED but still DEBT.**
      corr 0.999995 -> 1.000000, |x|ratio 0.999958 -> 1.000001 (|ratio-1| = 5.5e-7, PASSES the
      ratio bar) — the slope-N^2 live-ladder fix moved it, as predicted. BUT pointwise |rel|
      median is 1.061e-06, so it FAILS the 1e-9 per-element bar and correctly stays DEBT.
      NOTE: under the pre-2026-07-28 gate this would have flipped to a FALSE "AT BAR" — corr and
      ratio both clean with three orders of per-element error underneath, the bn2 pattern exactly.
      The per-element bar caught it automatically.
      Per-level err_norm is FLAT (max/min 1.1x over 35 levels) => NOT another depth-ladder
      problem; the remaining cause is unidentified. `aeiv` not measurable: eiv_dump_aeiv.bin does
      not exist in RUN_GDB (would need NEMO instrumentation).

## Medium Priority
- [ ] `traadv_fct` family (~1e-4/1e-5), `ATF filter u/v`, `dyn_drg_init`, `dyn_ldf u/v`,
      `dyn_vor EEN u/v`, `dyn_cor_2d`, `zdftke pdlr`/composite.
- [ ] The near-bar rows: `ssh_nxt` (4.0e-6), `dom_qco_r3c r3t` and `r3u/r3v` (3.0e-6).
- [ ] `lbc_lnk sign` — UNMEASURED and never verified. Cheap; just undone.
- [ ] `mlf_baro_corr` — UNMEASURED; needs a `_step_impl` diagnostics hook before it can be
      measured at all. NEMO dumps already exist.

## BLOCKED — escalate to the human, do NOT decide these
- [ ] `dyn_hpg dv` — deferred v-face metric. Genuine design tension with the #516 invariants.
      NOTE: the analogous `dy_v` worry was a FALSE DILEMMA (see the task file); check whether this
      one is too, and if it is, say so with evidence rather than assuming symmetry.
- [ ] `zdf_mxl_turb` — legoESM has NO equivalent. Implement, or waive explicitly (diagnostic-only
      in NEMO, never feeds dynamics). The waiver is a human call.
- [ ] `STABILITY on NEMO true grid` — from rest 5 years are stable, but the RESTART-start blow-up
      (max|u| 0.66 -> 3 m/s over 20 d) is real and unfixed. Open-ended.
- [ ] Anything that would change non-fidelity recipes' numerics.

## Completed 2026-07-28
- [x] `eos_rab alpha` (0.0), `eos_rab beta` (0.0), `bn2` (1.41e-17), `zdf_mxl nmln` (0/9920),
      `zdf_drg_nonlin` (0.0) — the first rows PROVEN per-element rather than on a cancelling mean.
- [x] Production fix: live z* EOS depth in `gm_redi_density_and_jacobian` (9156aa13e).
- [x] Production fix: same defect, slope-N^2 call site (1b37ea059).
- [x] Production fix: `zdepu`/`zdepv` face averaging; the v-slope was being fed `zdepu` (7bdabe4be).
- [x] Gate hardened: per-element bar + binary-gate encoding + non-vacuity self-test.
- [x] Mechanical gates: `precision_gate.require_fp64`, `time_levels.time_level_for_dump`.
- [x] `nemo_isotropic` extended to `dy_v`; all 9 #516 invariant tests still pass.

## Notes
- ONE row (or one shared cause) per loop. Depth over breadth.
- Every claim in a commit message must be a number you produced this loop.
- Update this file each loop: tick what closed, and append what you MEASURED (including negative
  results — a falsified hypothesis saves the next loop hours, and this campaign has already burned
  weeks re-testing dead ideas).
