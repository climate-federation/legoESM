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
- [ ] **THE BAROTROPIC SEED — now the single highest-value target (3 rows).**
      MEASURED 2026-07-29 (spg_substep_chain.py): `dyn_spg_ts` is NOT the defect. The velocity
      error is already at the LOOP-ENTRY SEED (err_norm u 2.31e-2) and flat through substep 1
      (2.31e-2) to final (3.09e-2). Substep count matches exactly (23==23, 68==68); alignment
      sharp at (0,0) on both a U-face and a T-point field; ssh EXACT at seed => velocity-specific,
      not shared with the eta/PGF chain. `puu_b`, `un_adv` and `pssh` are ALL INHERITED from the
      seed — the solver faithfully propagates a wrong initial condition.
      **RESOLVED TO (B) 2026-07-29: the AVERAGING OPERATOR owns it; the 3-D velocity is
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
