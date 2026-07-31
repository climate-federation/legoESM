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

## HUMAN DECISIONS 2026-07-30 (the three BLOCKED items are DECIDED — see task file for full text)
- [ ] `dyn_hpg dv` → NEMO-match, RECIPE-GATED. Extend `nemo_isotropic` to the dynamics v-face
      metric on NEMO cards only; row credited ONLY after all #516 adjointness+conservation tests
      pass UNDER the NEMO convention (machine-gated, same PR). Queued in work-order step 4.
- [ ] `zdf_mxl_turb` → VERIFY-UNCONSUMED, then WAIVE. Grep DINO-active NEMO source proving
      nothing consumes hmld/mldkz5; record an explicit HUMAN-WAIVED gate row with reason + grep
      evidence. A found consumer voids the waiver → escalate. (Cheap; interleave anytime.)
- [ ] `STABILITY on NEMO true grid` → FOLD INTO THE EIV ROW. eiv's exit criterion includes the
      20-d restart-stability re-test on e3t=both (max|u| holds ~0.6, not ->3). Exact-eiv-still-
      blows-up → PAUSE + ESCALATE.
- [ ] STILL ESCALATE: anything changing non-fidelity recipes' numerics; any row where the exact
      NEMO transcription is in place and the row still misses the bar (THE RULE, 2026-07-30).

## THE RULE (human, 2026-07-30): NEVER GUESS — transcribe from NEMO source with file:line in the
## commit; if an exact match does not clear the row or the behaviour, PAUSE the row and ESCALATE.

## ★ WORK ORDER REORDERED 2026-07-30 BY MEASURED MAGNITUDE — see the task file for the full text.
## The ACC acceptance run (88j) measured the 3 landed fixes as CLIMATE-INERT (upper contrast
## 0.9082 -> 0.9082 BIT-FLAT at y1 AND y2; dACC -0.005 Sv vs a 25.6 Sv deficit, 4 orders short).
## They change the state but in the WRONG PLACE (upper ocean, north of the band; the deficit is
## below 1000 m in the southern channel). ⇒ 1e-6..1e-5 rows do NOT move the ACC.
## NEW ORDER: (1) dyn_spg_ts puu_b 1.3e-2 + un_adv 8.4e-3 [the zu_frc/zu_trd absence thread],
## (2) eiv transport v 9.5e-3 / u 3.9e-3, (3) ZAD 4.9e-3 / ATF 4.4e-3 / dyn_ldf 3.9e-3,
## (4) zdftke DEMOTED (formula already exact; finish the in-flight sh2 face-native fix first),
## (5) everything <1e-3 + cheap interleaves LAST.
## ★ TRUE-LADDER INSTABILITY PROMOTED to a first-class lane: it BLOCKS the only clean acceptance
## test (baseline must run e3t=off, the 1-D ladder that is 12.9% wrong below k=25 — exactly where
## the deep fixes are meant to act). Deep fixes cannot be honestly evaluated until it is fixed.

## WORK ORDER (human-approved 2026-07-30 — SUPERSEDES the ordering above where they conflict)
1. `zdftke composite` (3.8e-3) + `zdftke pdlr` (3.2e-3)   <- CURRENT (iteration 2026-07-30a)
2. `eiv transport u/v` (9.5e-3/3.9e-3) + true-ladder stability exit test
3. `ATF filter u` (4.4e-3), `dyn_adv ZAD` (4.9e-3)
4. `dyn_spg_ts puu_b`/`un_adv` (the zu_frc/zu_trd thread above), `dyn_hpg dv` per decision
5. cancelling rows + `lbc_lnk sign` + `mlf_baro_corr` + `zdf_mxl_turb` waiver: interleave
NOTE: the zu_frc/zu_trd lead (High Priority above) stays QUEUED at 4, not dropped — it is the
most advanced thread and the `zu_trd` ABSENCE (dynspg_ts.F90:304/:367) is a pure transcription.

## 2026-07-30a iteration log
- Gate at start: `AT BAR 11 | DEBT 31 | UNMEASURED 3 | total 45` (verbatim, reproduced).
- [x] **stpmlf.F90 call-graph coverage table SHIPPED** (commit 2ee73f5ac,
      stpmlf_call_coverage.py + test, `6 passed in 0.18s`): 106 CALLs enumerated from DINO's
      OWN override (cfgs/DINO/MY_SRC/stpmlf.F90, key_qco branch), 28 COVERED / 69 WAIVED /
      **9 UNCOVERED — never measured, invisible until now**: wzv (x2 call sites), tra_zdf,
      dyn_zdf, traldf_iso_lap tendency, ldf_dyn coefficient, tra_qsr, ssh_atf, tra_sbc.
      Ranked [1] wzv + tra_zdf (judgment). NOTE tra_qsr = shortwave DEPOSITION (sets where
      heat enters the column — upper-stratification relevant); tra_zdf/dyn_zdf = the implicit
      solves the zdftke row feeds. Gate rows being added (agent in flight) so these count as
      UNMEASURED debt honestly.
- [x] **zdf_mxl_turb waiver evidence COMPLETE, waiver SUPPORTED**: hmld is output-only
      (zdfmxl.F90:152-158); only arithmetic consumers are PISCES via oce_trc.F90:93 and DINO
      compiles NO TOP tree (no key_top in cpp_DINO.fcm; no oce_trc.F90 in cfgs/DINO/WORK/).
      Gate row -> HUMAN-WAIVED: DONE (commit b9d4cb4c1). Human decision 2 of 3 fully executed.
- [x] **Gate absorbed both results (commit b9d4cb4c1, `12 passed in 0.19s`): new line
      `AT BAR 11 | DEBT 31 | UNMEASURED 10 | WAIVED 1 | total 53`** (was 11/31/3/45; the 9
      uncovered entries = 8 rows, wzv's two call sites share one). WAIVED never passes exit;
      waiver requires decision+evidence strings (3-variant synthetic-violation tested);
      WAIVED_ROWS is a closed dict guarded by test_no_generic_waiver_mechanism. The gate also
      gained its first pytest wrapper (test_fidelity_bar_gate.py — was CLI self-test only).
- [x] **zdftke instrumentation DONE (commit d635b99e0, `11 passed in 0.36s`).** 5 new dumps
      (units 8887-8891, MY_SRC/zdftke.F90, kt=57601, all 2,980,224 B = 52x199x36 f64, all
      registered with citations): tke_dump_{en,zmxlm,zmxld,avt_final,avm_final}.bin. Build
      "Compilation successful"; bit-identity control PASS (re-dumped rn2b md5 identical,
      cmp zero diffs). ACTIVE namzdf_tke recorded (only nn_etau=1 set by DINO; rest ref
      defaults incl. ln_lc=.true. rn_lc=0.15, nn_htau=1, nn_bc_surf/bot=1).
      **KEY DISCOVERY — the `zdftke composite` row's reference is CONFLATED: dump_avt/avm.bin
      are written in ldftra.F90:902-903, ONE SUB-STEP LATER = post-zdf_evd/zdf_ddm composites,
      NOT the TKE-closure output. The 3.8e-3 may not be zdftke's at all.** avt_final/avm_final
      (true zdftke exit) now let the split be measured on the NEMO side alone.
      **Chain corrections (verified, cite zdftke.F90): pdlr computed in tke_tke BEFORE the
      solve, from rn2b (BEFORE N2); zmxlm uses rn2 NOW (:740); nn_etau penetration INSIDE
      tke_tke; zmxlm=MIN(lup,ldown), zmxld=sqrt(lup*ldown).**
- [x] **WALK DONE (commit b7872175b) — TWO CONFIRMED TRANSCRIPTION BUGS, both verified by an
      independent single-column manual NEMO transcription reproducing the dump to 4-6 sig figs:**
      Stage table: entry 0.0 | pdlr 0.01096 (11 cells) | EN unmeasured (absence, below) |
      **MXL zmxlm 0.15914 / zmxld 0.21229 (331k cells) <- FIRST JUMP, owns the row** |
      avt/avm isolated 0.0 / 6.5e-6 | composite 0.9991 (86.7% same-step-EVD-fired).
      **BUG 1 (MXL): tke.py:698-701 mxl_choice==3 ldown scan seeds carry from lT[-1] instead of
      NEMO's zmxlm(jpk) init (zdftke.F90:678-679, read at :786-789). Near-neutral deep column:
      lego l_k=3454.8 m vs NEMO 617.5 m. Bites ONLY near-neutral deep columns — the southern-box
      regime; PLAUSIBLE direct link to the ACC deep-stratification destruction (5.6x mixing
      length there -> inflated deep avm/avt). Shallow/stratified unaffected (why spot checks
      missed it).**
      **BUG 2 (Prandtl): tke.py:1324 nemo_ri clamps max(sh2+bshear,1e-30); NEMO zdftke.F90:469-474
      takes zdiv AS-IS with a sign condition — negative zdiv => pdlr=1.0, lego gives 0.1. 11
      cells, each traced.**
      Stage-5 near-exactness (fed NEMO's own zmxlm/pdlr) => BUG 1 is the plausible owner of BOTH
      zdftke gate rows once production-wired.
      **COMPOSITE ROW CONFLATION QUANTIFIED: 86.7% (65313/75311) of avt_final-vs-dump_avt diff
      cells are same-step EVD-fired — the row's reference measures EVD+DDM, not the closure.
      Re-point it at tke_dump_avt_final/avm_final after the fix lands (separate reviewed step).**
      **EN-stage ABSENCE: rn2(now) and the SAVE'd dissl carry are not dumped — one more small
      instrumentation pass needed before the solve stage can be measured. Queue next iteration.**
      Registry catch: 4 previously-dumped-but-unregistered Prandtl dumps now registered
      (fail-closed design worked).
- [ ] **CURRENT: FIX both bugs (agent in flight)** — pure transcription per THE RULE, tests
      shown failing pre-fix, bit-identity proof for non-NEMO paths (mxl_choice==2, other
      prandtl modes), double-where AD safety on the sign condition, adversarial review, then
      production-wired re-measure of both gate rows. Gate-row number updates = separate step.

## 2026-07-30b iteration log (parallel lane — zdftke row untouched, its fix agent in flight)
- Gate at start: `AT BAR 11 | DEBT 31 | UNMEASURED 10 | WAIVED 1 | total 53` (verbatim).
- [x] **DONE (commit e34598cf9): 5 rows MEASURED, 4 confirmed-unmeasurable with exact reasons.
      New gate line `AT BAR 12 | DEBT 35 | UNMEASURED 5 | WAIVED 1 | total 53`** (`12 passed`).
      Measured: `lbc_lnk sign` EXACT (BINARY True; DINO zonally re-entrant, no north fold —
      psgn consumed only by the dead north-fold branch, lbc_lnk_pt2pt_generic.h90; the ~4e-6
      on stp_dump_07 momentum = DUMP-TIMING harness artifact, predates finalize_lbc — proven
      exact on atf_dump_uu/vv_before); `ldf_dyn` ahmt AT-BAR-quality / ahmf DEBT (ratio
      1.00001399, median rel 1.86e-5); `ssh_atf` near-bar DEBT (7.08e-7); `tra_sbc` DEBT
      (tem ratio 0.99993829, sal 0.99995924); `tra_qsr` DEBT (0.99996-class).
      Unmeasurable (exact needs recorded in-row): wzv (needs a ww dump), tra_zdf (needs
      GM-vertical+z*-volume port), dyn_zdf (needs implicit-bottom-drag-fold port),
      traldf_iso_lap (no MY_SRC bracket exists).
      **★ TWO STRUCTURAL ABSENCES FOUND (transcribe per THE RULE, not hypotheses): legoESM's
      DINO surface forcing LACKS NEMO's `emp*T*rcp` heat-content term (-> tra_sbc residual)
      and ssh_atf's emp-forcing-removal correction. BOTH are surface terms acting from day 0 —
      exactly the "upper contrast flat-wrong FROM YEAR 1" ACC signature. With the MXL
      deep-mixing-length bug (30a), the sweep now has one candidate per half of the
      stratification crossover: surface absence (upper, too sharp) + deep mixing-length bug
      (deep, destroyed). Neither yet proven to move the ACC — acceptance metrics after fixes. ★**
- [x] **zdftke FIX LANDED (commit e0fac585e, 4 files, review round 1 ISSUES-FOUND -> round 2
      CLEAN).** Bug 2 (Prandtl nemo_ri): full 3-way NEMO branch (zdftke.F90:459-476) via
      double-where; production-wired **corr 1.0000000000 / ratio 1.0000000000 / max|rel|
      4.4e-16** — the literal bar. Bug 1 (MXL choice-3 ldown seed, zdftke.F90:678+786-789):
      walk err 0.159->0.0099 / 0.212->0.0137 (16x), pathological bottom row max|diff|=0 vs
      NEMO dump; transcription CONFIRMED exact (independent reviewer port matches 1e-15).
      Pre-fix failures reviewer-reproduced; all non-NEMO paths bit-identical (8x 0.000e+00);
      3 broad-suite failures pre-existing (confirmed with fix stashed). Review caught a stale
      sibling test ENCODING THE OLD BUGGY SEED as truth (test_rk3_ws_and_mxl3) — fixed.
      **ESCALATION ITEM (THE RULE, designed outcome): MXL residual 0.0099 did NOT reach 1e-9.
      Transcription proven exact => residual PLAUSIBLY the walk's own rn2b-as-N2 proxy
      (rn2-now never dumped) + static-e3t harness inputs. RESOLUTION = the batched rebuild's
      rn2 dump. Do not tweak tke.py further.**
- [x] **Gate-row re-measure landed (ded827e12) + TWO CORRECTIONS ORDERED (agent resumed):**
      **pdlr ROW STAYS DEBT (0.997960/0.996970, barely moved) — CORRECTING MY OWN EARLIER
      "first row closed at the literal bar" claim: the FORMULA is exact in isolation
      (4.4e-16); the ROW is not closed.** Both e0fac585e fixes verified in production (the
      near-10x Pr signature is GONE), yet the row's ~2e-3 residual persists =>
      **UNIDENTIFIED upstream cause at the production call site's INPUTS (sh2/rn2b/avm_in —
      candidate: avm_in time level, NEMO feeds PREVIOUS-step avm). Next walk target.**
      Agent's "MXL bug not touched by e0fac585e" attribution REFUTED mechanically (git show:
      the ldown-seed fix IS in the commit) — correction 1 sent.
      Composite row: re-point correct, but its new tuple (0.4675/986.9) compared legoESM's
      POST-EVD production K_v vs NEMO's PRE-EVD closure exit = the avt_k trap INVERTED
      (ratio ~987 = EVD-100 cells vs closure ~0.1) — correction 2 sent: comparand must be
      legoESM's CLOSURE-ONLY avt/avm (southern_vmix_profile.py ablation-spy pattern).
- [x] Reviewer's independent final sweep CONFIRMS CLEAN: `2 failed, 341 passed` (4553
      deselected), both failures accounted for — the peer-fixed stale-seed test (re-verified
      `1 passed`) and the pre-existing `test_solver_dz_cell_without_dz_surface_raises`
      stale regex (fails on HEAD too, unrelated).
## 2026-07-30c iteration log
- Gate at start: `AT BAR 12 | DEBT 35 | UNMEASURED 5 | WAIVED 1 | total 53` + NEW FLAG:
  1 of 12 AT BAR rows is on CANCELLING statistics only (per-element never measured — the
  bn2-class gap). Identify which row + measure per-element: cheap interleave, queued (e2).
- [x] **Surface absences: BOTH RETRACTED (commit 536ae1d3f — record corrected, no code
      forced).** tra_sbc emp term is `lk_linssh`-gated (FALSE under key_qco) AND **DINO's
      emp ≡ 0 identically** (CASE(4), ln_emp_field=.false. → usrdef_sbc.F90:254-259
      `emp=0._wp` unconditionally; runtime-confirmed ocean.output). ssh_atf correction:
      gate fires but zwght = 0−0 = 0 exactly. The probe had cited CASE(5) code DINO never
      runs. Reviewed CONFIRMED; probe re-run bit-identical. **The "surface absence = year-1
      ACC signature" candidate is DEAD; the ACC upper clue is unexplained again.**
      **★ REPLACEMENT LEAD (PLAUSIBLE, uninstrumented): surface-flux DIVISOR — NEMO divides
      by LIVE `e3t(:,:,1,Kmm)` (trasbc.F90:152-153, z*-varying); legoESM uses constant
      `dz_ref[0]`. STATIC-VS-LIVE LADDER FAMILY, 4th site. Probe then (if confirmed) fix =
      the tra_sbc row's next step. Also grep ALL other surface-forcing divisor sites while
      there (the standing every-call-site rule).**
- [x] **Corrections landed (323616372).** pdlr row provenance now carries the retraction +
      the input-by-input evidence: **sh2 corr 0.9344 / ratio 0.9601 (THE BAD INPUT)**, rn2b
      1.000000, avm_in cell ratios 1.04-1.10. Composite row re-measured closure-vs-closure
      (ablation spy, non-vacuity max|K_prod-K_closure|=100): 0.9666277701/1.0708326784 avt,
      0.9671/1.0649 avm (offset near-tie caveat recorded). Both DEBT, honestly.
## 2026-07-30k iteration log (first under the tightened token economy)
- Gate: `AT BAR 15 | DEBT 32 | UNMEASURED 5 | WAIVED 1 | total 53`. Lanes clear, GPUs free.
- [x] **ZAD row RECORDED (commit `5c441e312`)** with the POPULATION stated explicitly — this
      row is the campaign's clearest case for why that matters. UNION (surface u-mask
      broadcast, includes below-seafloor cells NEMO discards at `dynzdf.F90:121`):
      corr 0.999660 / ratio 0.999662 / err 2.6066e-02. **ACTIVE-ONLY (full 3-D umask = what
      NEMO keeps): corr 0.9999999996 / ratio 1.0000015 / err 2.854e-05 — RECORDED.**
      PER_ELEMENT 3.278e-05. `classify()`: corr CLEARS (1-corr 4.06e-10), ratio misses by
      1.5x, per-element by 3.3e-5 ⇒ **stays DEBT, near-bar. Overall 4.9e-3 -> ~3e-5, >100x —
      the campaign's first LARGE fidelity gain.** Stale self-check in
      `zad_level29_onset_walk.py` repaired (post-fix asserted, pre-fix kept as comment).
- [x] **ECONOMY IMPLEMENTED + DURABLE**: cron 3-hourly (FLOOR; human fires bursts);
      one-lane rule + scope caps in BOTH the cron prompt and the task file; digest at
      `docs/ocean/fidelity/dino_1226_state.md` (93286444c). Measured basis: ~30 lanes x
      100-400k = ~95% of spend. PROTECTED: adversarial reviews; numeric precision in
      compression.
- [ ] **IN FLIGHT (one capped lane): is `zdftke composite` INHERITED from `sh2`?**
      Chosen by MAGNITUDE (the acceptance-derived priority): its closure-vs-closure ratio
      **1.0708** (~7%) is the LARGEST relative error left, and unlike `zu_frc`/`sh2` it is not
      blocked behind a pending human decision. Brief capped to steps 1-2 then STOP:
      (1) the `sh2 -> en -> avm = rn_ediff*zmxlm*sqrt(en)` chain with file:line **plus the
      magnitude arithmetic — note the SQRT halves a fractional en error, so if sh2's ~10%
      restricted residual cannot produce ~7% in avm, inheritance dies on magnitude alone**;
      (2) substitute NEMO's own dumped `sh2` into legoESM's closure, one variable, re-measure.
      Collapses => INHERITED, tie the row to ESCALATION 1 rather than chasing it separately.
      Does not move => INDEPENDENT, name where the chain first diverges and stop.

## 2026-07-30j iteration log
- Gate: `AT BAR 15 | DEBT 32 | UNMEASURED 5 | WAIVED 1 | total 53`. Lanes clear, GPUs free,
  production tree clean.
- [x] **`dyn_ldf` 35x contradiction RESOLVED (commit `43f1ad2ed`): Measurement A wins, NO
      RETRACTION.** B's probe omitted the MANDATORY `mask=` kwarg — NEMO folds `fmask` into
      `ahmf` at init (`ldfdyn.F90:330`; `dynldf_lev_rot_scheme.h90:23` says so in its own
      comment), and legoESM's coefficient fn returns a pure 1-D latitude array that CANNOT
      carry it, so the caller must. Production does; B didn't.
      **★ THE 2x2 IS THE REAL OUTPUT — SYNTHETIC INPUTS HID A MANDATORY-MASK BUG:
      synthetic 0.9649 -> 0.9673 with the mask (barely moves) vs REAL DATA 0.7060 ->
      0.999999999 (decisive). ⇒ NEW HARNESS RULE: dynamic range and boundary coverage are
      DIFFERENT properties; synthetics only buy the first. Anything mask/wall/seam/land
      sensitive MUST be cross-checked on REAL restart data.★**
- [ ] **IN FLIGHT: `dyn_adv ZAD` — the two surviving candidates.** (1) the `zWdzU` RECURRENCE /
      vertical-flux structure (k-ordering, boundary seeds at jk=1 and jk=jpkm1; a vectorised
      gather equals a recurrence ONLY if direction and seed match — precedent: `ldfslp.F90:209`
      `DO jk = jpkm1,2,-1` where it DID); (2) the s-zone `e3` GRADIENT (`|e3(k+1)-e3(k)|/e3(k)`)
      — a DIFFERENT quantity from the divisor VALUE, which is already refuted by A/B.
      Magnitude arithmetic MANDATORY: any mechanism must PREDICT ~4-5e-2 in jk 30-34.

## 2026-07-30i iteration log
- Gate: `AT BAR 15 | DEBT 32 | UNMEASURED 5 | WAIVED 1 | total 53` (was 13/34 — ATF u/v closed
  at the bar). All lanes clear, GPUs free, production tree clean.
- [ ] **IN FLIGHT: reconcile the TWO `dyn_ldf` measurements, ~35x apart, both from today.**
      A (`ww_inheritance_walk.py::measure_dyn_ldf_corrected`, `f51cb3318`, RECORDED): Kbb-fed
      err_norm u 4.5769e-05 / v 4.5052e-05 -> corr 0.999999999 / ratio 1.000001864.
      B (`unit_harness/run_dynldf_lap_probe.py`, `e08132968`): corr ~0.965 / ratio ~1.034,
      localized to the periodic-seam u-face wall convention; that agent correctly REFUSED to
      force it into the gate.
      **CHOSE THIS OVER THE HIGHER-LEVERAGE ZAD ROW DELIBERATELY: an unreconciled contradiction
      is a correctness bug in our KNOWLEDGE, and this one tests a claim I have repeated all day
      — `dyn_ldf` is one of the six "harness artifact, not a model defect" successes. If B is
      right, that characterisation RETRACTS and part of today's headline reverses. Rule 1e
      says neither number may stand until reconciled.**
      Brief orders METRIC RECONCILIATION FIRST (rescore each side under the other's convention),
      because metric-mixing is a THREE-TIME repeat offender today — most recently the ZAD
      harness-vs-walk 5x gap, which was pooled-vs-per-level aggregation, neither number wrong.
      Only if a real numerical disagreement survives does it hunt physics, starting at the
      periodic seam (DINO is zonally periodic / meridionally walled; stored columns 0 and
      n_lon-1 are ONE cell apart) and the `A2D(0)` explicit-shape extent hazard.

## 2026-07-30h iteration log
- Gate: `AT BAR 13 | DEBT 34 | UNMEASURED 5 | WAIVED 1 | total 53`. GPUs free. HEAD `0f9f008d4`.
- **ORCHESTRATION NOTE (my error, avoid repeating): I granted BOTH the ZAD lane and the harness-
  scaling lane permission to edit `fidelity_bar_gate.py`. That is a collision risk. Only ONE lane
  at a time should hold the gate file; give the others measurement-only briefs.**
- [ ] **IN FLIGHT: `ATF filter u/v/T-S-ssh` walk** (work-order #3's remaining item, 4.4e-3,
      untouched, independent of the momentum reconstruction). MEASUREMENT-ONLY brief — no gate
      edits, no production fixes — precisely to avoid the collision above.
      **The brief front-loads the TIME-LEVEL AUDIT, because an Asselin filter is BY DEFINITION a
      multi-time-level operator and is therefore the single most likely place for the `dyn_ldf`
      class of error** (that row's entire 3.9e-3 DEBT was the probe feeding NOW where NEMO reads
      Kbb; production was already correct; err_norm 4.49e-2 -> 4.58e-5). Also required: the
      live-vs-static e3t answer under key_qco (5 confirmed ladder sites), per-level profile as
      the ladder discriminator, and a harness-artifact-class check per dump used (which side of
      any later operation it sits on, with the F90 write line).
      **If ATF is ALSO a harness time-level artifact that makes FIVE, and would mean the DEBT
      list substantially overstates the model's real infidelity.**

## 2026-07-30g iteration log
- Gate at start: `AT BAR 13 | DEBT 34 | UNMEASURED 5 | WAIVED 1 | total 53`; the NEW provenance
  check is live and naming rows (`sbc`, `ssh_nxt/div_hor`, `zdf_drg_nonlin`, `zdf_mxl_turb`, ...).
  Cancelling-only count down to 1 of 13 (`lbc_lnk sign`). GPUs free, production tree clean.
- [ ] **IN FLIGHT: the STOPPING-RULE audit of the four `ldf_slp` rows** (wslpi/wslpj/uslp/vslp).
      All four CLAIM the full CONDITIONING-LIMITED profile: every chain stage at roundoff
      (prd 2.559e-6 -> 1.804e-11, zbw 3.467e-07 -> 9.369e-16 after the two live-depth fixes),
      transcription verified line-by-line vs ldfslp.F90 (zcj :307-308, zbj MIN :317, zfk :320,
      the :209 `DO jk = jpkm1,2,-1` recurrence), and the amplification demonstrated (zbj p99
      7.067e-11 -> zww_raw p99 4.437e-04 through zaj/(zbj-eps); k=34 zbj 5.2e-9 / zaj 4.4e-5 /
      zww_raw 1.5e-2). **The agent must VERIFY each condition with its own numbers — a claim in
      a gate note is NOT a measurement, and this campaign has found several that did not
      reproduce.** If all three hold -> annotate CONDITIONING-LIMITED + ONE consolidated
      escalation (they share a cause), DEBT classification UNCHANGED. If any fails -> that row
      still has identifiable work and must NOT be escalated (the more valuable outcome).
      **This is 4 of 34 DEBT rows resolvable in one action — the largest single bookkeeping
      gain available, and legitimate ONLY if the evidence is real.**
- [ ] Same lane, cheap: measure a per-element statistic for `lbc_lnk sign` to drive the
      cancelling-only count to 0 (or record WHY per-element is structurally meaningless for a
      binary-identity row — no invented numbers).

## 2026-07-30f iteration log
- Gate at start: `AT BAR 12 | DEBT 35 | UNMEASURED 5 | WAIVED 1 | total 53` (the tra_sbc/tra_qsr
  recording agent had not landed yet — expect AT BAR 13+ once it does).
- HEAD `9286b8309`. 4 lanes live: gate-recording+probe-fix (a385748c...), shear face-native
  transcription (a8f5f1b5..., 5 dirty production files), zu_frc/zu_trd walk (aaeafa28...),
  acceptance run y5 (GPU, PID 2404290).
- [x] **c_p + qsr landed (9286b8309, review CLEAN): FIRST DEBT->AT BAR CLOSURE — tra_sbc tem
      9.657e-07 -> 1.936e-16 (both components at bar); tra_qsr 2.024e-05 -> 5.223e-07 (~39x).**
      c_p was a REUSE: `constants_config.py:56,67` already held NEMO's 3991.86795711963 cited
      to eosbn2.F90:1899; card precedent (`g`, `omega` as recipe overrides) settled placement.
      T/S asymmetry root: trasbc.F90:136-137 — tem carries r1_rho0_rcp, sal only r1_rho0.
      **9th blind-instrument instance found (SAME FILE, 2nd time): coverage_rows_measure.py is
      also blind to `shortwave_penetration_ladder`. LESSON: fixing one config-blindness in a
      probe does NOT fix its siblings — audit EVERY dispatch-carrying call when you find one.**
- [ ] **NEW LANE (work-order #2): eiv transport u/v walk** (aee60046..., eiv_transport_walk.py).
      **Tests the STANDING-BUT-NEVER-TESTED hypothesis: "static t_depth_ref vs live z-star gdept,
      deepest 1-2 levels" = the LADDER FAMILY'S 6th SITE.** Prior is strong (5 confirmed sites).
      Decisive cheap test FIRST = the PER-LEVEL residual profile: deepest-levels concentration
      CONFIRMS the shape, FLAT REFUTES it (flatness is exactly what ruled the ladder out for the
      `ldf_eiv aeiu` row, so it is a real discriminator). Then A/B static vs live, one variable.
      u-vs-v asymmetry (9.5e-3 vs 3.9e-3, ~2.4x) to be treated as a CLUE, not assumed shared —
      the campaign has been bitten assuming u/v symmetry (zv_frc is 15x smaller than zu_frc for
      a genuinely different reason).

## 2026-07-30e iteration log (ALL THREE ACCELERATIONS APPROVED BY HUMAN — enacted)
- Gate at start: `AT BAR 12 | DEBT 35 | UNMEASURED 5 | WAIVED 1 | total 53`.
- GPU 0 = my own shear-fix agent's pytest (PID 2381359, 24.7 GB) — DO NOT TOUCH. GPU 1 free.
- **(1) PARALLELISM WIDENED to 4 lanes** (credit cost is linear — hold at 3-4, beyond that
  wall-clock gains stall because review/recording serialize anyway):
  - [ ] A: probe-blindness fix (coverage_rows_measure.py never reads
        `cfg.surface_flux_divisor` — 8th blind instrument) + **the T/S ASYMMETRY walk**:
        why tem keeps per-element 9.657e-07 while sal clears to 4.4e-16 on the SAME fix.
        Prime candidate = the Q_sr shortwave path (T-only; and `tra_qsr` is independently
        DEBT at 0.99996-class) — test whether tra_sbc-tem and tra_qsr share ONE cause.
  - [x] B: **momentum Jacobian — SUSPECT CLOSED (commit 1dfd4d9b8).** My "global H_max"
        framing was WRONG (retracted): under e3t=both legoESM's J is already LOCAL,
        `(eta+H_bathy)/H_bathy` (vertical.py:824-833); the H_max formula is in the INACTIVE
        OceanZStarCoordinate branch. What remains is an order-of-operations difference and
        it is NEGLIGIBLE: median u-face thickness ratio **1.00000000** (passes the bar),
        max|ratio-1| 4.10e-05, divisor A/B err_norm 4.57e-06.
        **zu_frc CONNECTION REFUTED: r = +0.11** (zu_frc's 8.03e-3 reproduced exactly first,
        alignment sharp at offset 0 — validated instrument). zu_frc's cause stays OPEN.
        Probe caught its own off-by-one face slice (`[:, :n_lon]` vs the established
        `[:, 1:]`): fixing it moved r from 0.53 -> 0.11. **The uncorrected 0.53 would have
        looked like a real shared cause.** Face-slice convention = recurring trap.
        NEW INSTRUMENTATION NEEDED (batched rebuild): a dump right after `dynzdf.F90:334-335`
        — no existing dump brackets dyn_zdf's surface stress alone (the stp_dump bracket
        folds in implicit bottom drag).
  - [ ] C: shear-production transcription (in flight from 30d).
- **(3) ACC ACCEPTANCE RUN LAUNCHED on GPU 1** (lane D) — from-rest 5-yr twin at HEAD with
  the 3 landed fixes (Prandtl sign, MXL seed, surface divisor); shear NOT in this run.
  **The task is PROTOCOL MATCHING, not the run**: baseline pinned field-by-field (recipe,
  frame, e3t mode, precision, dt, years, forcing, ACC metric incl. `compare_fullframe.py:18`
  e3t_1d weighting + median over lons 2..-2) with an instruction to STOP rather than run
  something plausible. Metrics: ACC 65.5 (NEMO 91.1); z(maxN2) 166.6 -> 110-127 m;
  **y1 upper contrast 0.908 -> 1.0 (SHARPEST — wrong from year 1, responds fast)**;
  deep contrast 0.86 -> 0.44. Verdict must be (a) moved toward / (b) inert / (c) moved away
  (= compensating error removed, a SIGNAL not a revert trigger), with nulls NOT rounded up.
- **(2) BATCHED NEMO REBUILD: correctly BLOCKED** — it regenerates dumps that lanes A/B/C are
  reading; firing now would silently corrupt live measurements. Fires when they land.
  Contents: rn2-now + dissl (EN stage + the MXL residual attribution escalation), ww (wzv
  row), traldf_iso_lap bracket, + any dump lanes A/B name as missing.

## 2026-07-30d iteration log
- Gate at start: `AT BAR 12 | DEBT 35 | UNMEASURED 5 | WAIVED 1 | total 53` (+ the
  cancelling-only flag on 1 of 12). All lanes idle, GPUs free -> ran the two independent
  top-queue items IN PARALLEL (disjoint files: zdftke shear vs surface forcing).
- [ ] **IN FLIGHT A: sh2 walk** (sh2_walk.py) — NEMO zdf_sh2 formula w/ file:line, term-by-term
      alignment table, A/B one factor at a time vs tke_dump_sh2.bin. Candidate classes to
      confirm/exclude BY READING: velocity time level (Kbb/Kmm mix), the vertical metric
      divisor (live e3uw(Kmm) vs static — THE LADDER FAMILY, 3 confirmed sites already),
      w-point averaging stencil + mask, avm factor's time level. Measurement only; a defect
      gets a separate transcribe+test+review task.
- [x] **Surface-flux divisor: CONFIRMED + FIXED (5e9b0eb87) + RECORDED (fbe04d93a).**
      `DINOConfig.surface_flux_divisor='nemo_live'` on the kamm cards (resolution PRINTED,
      not assumed); shared `nemo_r3t_stretch` in eos.py; default bit-identical; reviewed SHIP.
      tem 0.99999985/0.99993829 -> 1.00000000/1.00000100; sal -> 1.0/1.0 per-element 4.4e-16.
      **⚠ ROW STAYS DEBT: tem per-element 9.657e-07 (3 orders over bar). I had claimed "at the
      bar" — RETRACTED, that 0.0 was SALINITY only. The per-element bar caught my prose error.**
      **NEW TARGET: the T/S ASYMMETRY — why does tem keep 9.66e-07 while sal clears on the
      same fix? Candidates: the Q_sr shortwave-penetration path (T-only), the tem-vs-sal
      restoring formulation, a T-only flux-construction term. Walk it like sh2.**
      **IMPLEMENTATION LESSON: a divisor change inside an IMPLICIT solve is NOT a post-hoc
      rescale — the first attempt rescaled the output and was 1.2-3% wrong because
      (tau_T + dt) is not proportional to 1/dz_0. Thread the live dz_0 into tau_T/tau_S
      BEFORE the implicit formula.**
- [ ] **PROBE DEFECT — fix before the next tra_sbc measurement:** `coverage_rows_measure.py`
      hardcodes `dz_0 = z_coord.dz_ref[0]` and never reads `cfg.surface_flux_divisor`, so it
      silently measures the STATIC path whatever card it is given (production dispatches
      correctly). 8th blind-instrument instance. Make it read the config.
- [ ] **NEW momentum gap (recorded UNMEASURED on the dyn_zdf row):** `surface_stress_faces`
      (ocean_pe_latlon_cgrid.py:3445-3481) uses `dz_0_T = dz_ref[0]*J`,
      `J = (eta+H_bathy)/H_max` GLOBALLY normalized; NEMO uses `r3u = eta/hu_0` LOCALLY
      normalized at the **Naa** level (`stpmlf.F90:305 CALL dyn_zdf(kstp,Nbb,Nnn,Nrhs,uu,vv,Naa)`
      — Naa, NOT Kmm; my brief said Kmm and the review refuted it). Differs on SLOPED
      BATHYMETRY — the same geometry+region where the zu_frc residual was localized
      (Stage-7 note). Momentum path -> feeds the largest barotropic DEBT rows. HIGH INTEREST.
- [x] (superseded) **IN FLIGHT B: surface-flux divisor probe** (surface_flux_divisor_probe.py) — the
      ladder family's 4th candidate site. Tasks: quantify (e3t(1,Kmm)/e3t_0(1) - 1) over wet
      cells and check the arithmetic against the ~6e-5 row residual; table EVERY legoESM
      surface-divisor site (not just the flagged one — the every-call-site rule); A/B
      constant dz_ref[0] vs live e3t(1,Kmm) against NEMO's dumped tendency, controlled
      one-variable, self-check = bit-identical when r3t forced 0. CONFIRMED or REFUTED.

- [ ] **(superseded by IN FLIGHT A above) WALK sh2.** NEMO zdf_sh2 (zdfphy.F90:268, called
      from stpmlf via zdf_phy) builds shear production from a SPECIFIC Kbb/Kmm velocity
      time-level mix — read the routine, compare legoESM's production sh2 construction
      (the spy point's sh2) term by term. sh2 feeds pdlr (via Ri) AND the en solve => one
      cause may close both remaining zdftke residuals (pdlr ~2e-3, composite 0.966/1.07).
      avm_in 1.04-1.10 = secondary suspect (time level: NEMO feeds PREVIOUS-step avm).
- [ ] THEN: (a) ONE batched NEMO rebuild (BLOCKED until the surface-absences agent lands):
      rn2-now + dissl + ww + traldf_iso_lap bracket (+ any sh2-walk needs discovered above —
      batch them); re-walk MXL with true rn2 (expect ~1e-9 or real escalation); (c) eiv row +
      stability exit test; (d) ACC acceptance run once zdftke+surface fixes are in;
      (e1) trivial: stale regex test_solver_dz_cell_without_dz_surface_raises; (e2) the
      cancelling-only AT BAR row's per-element measurement.
- [ ] Stale cross-reference noted by the gate agent, cheap cleanup later: dangling "see the
      UNCOVERED entry below" comment near r3f (~line 253 of stpmlf_call_coverage.py).

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
