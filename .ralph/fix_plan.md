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
- [ ] **Finish the metric_convention pass (human APPROVED: go for the NEMO match).**
      `nemo_isotropic` already sets `dy_T`/`dy_u`/`area_T` and now `dy_v = R*dlon*cos(lat_v)`
      (NEMO `pe2v = pe1v`, `usrdef_hgr.F90:117`). REMAINING: wire the DINO bridge to select it
      (`bridge_nemo_to_legoesm_topo` defaults to `"exact"` regardless of the recipe card — a
      harness gap), then ONE controlled pass recording before/after for EVERY affected row
      (`vslp`, `wslpj`, `zaj`, `ssh_nxt`, `dyn_cor_2d`, anything else that moves).
      Then adversarial review — this touches barotropic v-point areas `1/(dx_v*dy_v)`, the PGF,
      and `latlon_cgrid_operators`.
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
- [ ] **Instrument NEMO for the big rows.** `dyn_spg_ts` (1.3e-2, the worst), `zdftke`, `dyn_vor`,
      `traadv_fct`. Add `WRITE` dumps of the INTERMEDIATES following the
      `cfgs/DINO/MY_SRC/ldfslp.F90` pattern (units 8840-8848, `l_1226_raw_dump_done` first-call
      flag), rebuild, rerun, register each dump in `legoesm.ocean.fidelity.time_levels` with its
      `file:line`. This is the single highest-leverage action: `ldf_slp` closed in hours BECAUSE
      its chain was dumped, while the 3e-6 band resisted for weeks on endpoints alone.
- [ ] **Suspect the static-vs-live depth-ladder defect at every remaining call site.** It has been
      found TWICE (`gm_redi_density_and_jacobian`, `_nemo_wpoint_e3w_wmask_n2`). Grep for
      `t_depth_ref` / `z_full_ref` / `cumsum(dz)` consumers and check each against NEMO's
      `gdept(Kmm)`. (This is an inference, not a measurement — treat as a prior, not a fact.)
- [ ] `vslp` residual after the metric pass, if any remains.
- [ ] `ldf_eiv aeiu` — re-measure; it shares `_nemo_wpoint_e3w_wmask_n2`, so the slope-N^2 fix
      should already have moved it off its recorded 0.999995 / 0.999958.

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
