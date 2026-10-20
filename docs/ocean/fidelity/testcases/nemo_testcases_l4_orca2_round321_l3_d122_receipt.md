# ORCA2 round 321 — lane B's L3 eddy-induced velocity: passes Decision 122 on rung 10 but leans hard away; held off

Date: 2026-10-20 (Claude round; codex review: operator).

Status: **STOPPED_FOR_DECISION** (import landed default-off; selection held).

## Verdict

- L3 imported default-off; every flag-off step program and rung 10's
  forward bits are unchanged.
- On rung 10 (the only card whose NEMO record runs it) L3 passes Decision
  122 (c)+(d) on all seven pairs, but leans hard away: census 2 toward /
  177 away (independent), 0 / 179 (given entry).
- Rung 10's kt-10 max temperature error would go 0.202 -> 3.002 K and max
  salinity error 0.143 -> 7.141. Not selected; decision below.
- Gradients with L3 on stay finite (0 non-finite), reverse = forward.

## Provenance (commits, this branch)

- `fcf91db15` import. Package hunks of gridladder-vmix
  `b3e5011190..fb02c4b29f` (L3 = `698ee4d7d8..fb02c4b29f`, after lane B's
  G9 merge; the merge's own package changes were already here from
  round 318), `git apply --3way`. Conflicts: the vertical module's
  signature takes lane B's explicit north-fold / ratio-exchange arguments
  (no caller here passed the old fold-grid argument; every caller checked);
  the stage-3 finaliser takes lane B's live Krhs tuple without lane B's
  BBL stage-3 arguments (not on this branch). Recipe: only the shared ORCA2
  isoneutral constructor hunks (eddy_induced_velocity, rn_Ue, rn_Le); the
  ladder-card hunks are not imported (no G/V/L rung tables here). One
  import line kept that lane B dropped (this branch still uses it).
- `1a9ad3c17` preregistration, gate, public L3 constructor (lane B's L3
  card arguments).
- `3c368f8ca` lane B's L3 unit-test hunks; `11d67e6b5` citation re-anchor
  and two new map entries.
- Gate: `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round321_l3_d122_gate.py`
  (round-285 ladder, round-318 later-first classifier = DECISION 139's
  safeguard, round-278 decision). CPU, fp64, libm, production JIT.
  Evidence `orca2_rounds/round321/` (`jaxpr/`, `probes/`, `arms/`, `grad/`).

## Flag-off identity

- Normalized step programs vs base `54a7595e0`: 11/11 identical (rung 0,
  OMT-0..4, rung 10, GYRE, VORTEX SMT flux and vector, DINO).
- Rung 10, three production steps: 72/72 leaf-steps bit-identical
  (round-316 forward-bits instrument).
- The flag-off L2 constructor returns the same config value as before.

## Which records run L3

Rung 10 only: its ocean.output states ln_ldfeiv = T, nn_aei_ijk_t = 21,
rn_Ue = 0.03 m/s, rn_Le = 200 km (EXP00 sets ln_ldfeiv at
`ORCA2_OMIP_L4/EXP00/namelist_cfg:317`).
Rung 0 and OMT-1..4 record ln_ldfeiv = F. The compiled statements are those
lane B certified: bolus into pFu/pFv (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:954-955`),
called at stage 3 (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:255-260`),
then ww recomputed (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:296`).

## Feasibility (trace only, before the preregistration)

| card | arm | result |
|---|---|---|
| rung 10 | L3 | traces |
| rung 10 | L3 + G9's stage-3 Krhs order | refuses: that order needs no BBL, and rung 10 runs the diffusive BBL (corrected at resume; rung 10 already runs the written content order) |
| rung 0 | L3 | refuses (same as L2 in round 320); its record runs no EIV |

## Decision 122, seven pairs (tree `1a9ad3c17`, arms control vs L3)

| pair | exact rows lost | first over-bar | census toward/away (early, late) | kt-10 max ssh before -> after (m) |
|---|---:|---|---|---|
| rung 0 | 0 | equal (kt 1 stage-1 T) | 0/0 | 0.00701981 -> same |
| OMT-1 (kt 8) | 0 | equal | 0/0 | 2.90886e-05 -> same |
| OMT-2 | 0 | equal | 0/0 | 4.30139e-05 -> same |
| OMT-3 | 0 | equal | 0/0 | 7.46458e-13 -> same |
| OMT-4 | 0 | equal | 0/0 | 1.89112e-04 -> same |
| rung 10 independent | 0 | equal (kt 1 entry ssh) | 2/177 (0/39, 2/138) | 0.183498 -> 0.183122 |
| rung 10 given entry | 0 | equal (kt 1 stage-1 T) | 0/179 (0/39, 0/140) | 0.168598 -> 0.168290 |

Decision: `PASS_R278_D122_DECISION`. Away > 2x toward on rung 10, so by the
preregistration it is asked, not landed.

## Rung 10, kt-10 stage-3 errors (|card - NEMO|, round 318's instrument)

| state | label | max ssh (m) | max T (K) | RMS T (K) | max S | RMS S |
|---|---|---:|---:|---:|---:|---:|
| shipped (L2) | independent | 0.18349806 | 0.20180847 | 3.5997e-3 | 0.14275824 | 2.2665e-3 |
| with L3 | independent | 0.18312161 | 3.0021558 | 1.0158e-1 | 7.1409592 | 7.1287e-2 |
| shipped (L2) | given entry | 0.16859805 | 0.20181174 | 3.5998e-3 | 0.14275574 | 2.2667e-3 |
| with L3 | given entry | 0.16828961 | 3.0021585 | 1.0168e-1 | 7.1409724 | 7.1246e-2 |

Velocity max errors also grow: u 0.154 -> 1.768, v 0.141 -> 1.722 m/s
(independent).

## Why it leans away (MEASURED by the Claude reviewer; reproduced at resume)

One rung-10 production step, `orca2_rounds/round321/review_claude_r321/`
(probe `probe_cont.py`), re-run on this tree at resume
(`orca2_rounds/round321/resume_repro/`):

- With L3 on, the horizontal transports the stage-3 FCT reads are
  bit-identical to L3 off; its vertical velocity includes the bolus.
- Subsurface continuity residual of those transports: 7.0e-5 m/s with L3
  (413,665 subsurface wet cells above 1e-15; the reviewer counted 409,337
  at its own threshold) vs 1.9e-17 with L3 off (0 cells). It equals minus
  the bolus divergence to 1.9e-17 (reviewer 3.9e-19). Lane B's card: 1.6e-17
  (reviewer).
- So the FCT moves tracer with a vertical bolus but no horizontal one; this
  is the cause of the away lean. CONFIRMED.
- Causal arm (reviewer only, SCRATCH DIAGNOSTIC, NOT A ROUTE: adds the
  bolus to the FCT mass-flux slots, which is not NEMO's order and must not
  be committed): residual back to 1.9e-17; kt-10 max T 0.202 K, RMS T
  3.31e-3 (control 3.60e-3), max S 0.143; per-row RMS 140 toward / 39 away
  (the reviewer's row comparison, not the D122 classifier).

## Predictions (PREREG_nemo_testcases_l4_orca2_round321.md)

- P1 completes, finite, both labels: CONFIRMED.
- P2 passes D122 (c)+(d): CONFIRMED.
- P3 lowers kt-10 RMS T (independent): FALSIFIED, 3.600e-3 -> 1.016e-1.
- P4 sea-level change under 20 %: CONFIRMED, -0.21 % (given -0.18 %).
- P5 gradients finite, reverse = forward: CONFIRMED (below).

## Gradients (rung 10, L3 selected, N = 1, `grad/grad_rung10_n1_l3.json`)

- T loss w.r.t. initial T: 0 non-finite entries; along one T direction
  reverse vs forward agree to 1.2e-14 (sea-level loss 4.0e-14).
- rn_Ud: reverse -4.5179e-05, forward equal to 3.0e-16; best FD relative
  error 3.5e-6 (step 0.01).
- aei0 (the eddy coefficient the step reads): reverse 2.7523e-06, forward
  identical; best FD relative error 2.8e-6 (step 1e-4). Not inert.
- Sea-level loss: both scalar gradients exactly 0 at one step.
- Traced and rebuilt production steps agree bit for bit for both scalars
  (0 leaves differ); a 1e-4 change moves 430,381 T cells (rn_Ud) and 601
  (aei0). Peak memory 68.5 GiB.
- No round-316 guard was touched by the import; shipped program unchanged,
  so round 320's shipped-card gradient result carries over.

## Round-314 audit rows

None change: rung 10 still runs no eddy-induced velocity (ln_ldfeiv row
stays DIFFERS, its three coefficient rows NOT-MODELLED).

## OPEN

- DECISION: L3 on rung 10. (1) keep it off until rung 10 runs G9's stage-3
  Krhs order (the route lane B certified), then re-measure; (2) land it now
  under D122 (passes; max T error 0.20 -> 3.0 K); (3) as (1), and also make
  the model refuse the L3 bolus without that route. Pick 1.
- G9's stage-3 Krhs order refuses on rung 10 because rung 10 runs the
  diffusive BBL (that order needs no BBL); rung 10 already runs the written
  content order (round 318). Trace check: rung 10 with BBL removed traces
  (`resume_repro/r10_trace_krhs_nobbl.log`). The Krhs order with BBL is the
  prerequisite for (1).
- DECISION (asked, not done): the guard refusing L3 without the stage-3
  Krhs route.
- DECISION 145 (rn_Ud trainable; kappa_Redi frozen on rung 10): held at
  resume, see Resume.
- Rung 0 runs no isoneutral diffusion although its record does (round 320).

## Gates (x64, CPU; logs `orca2_rounds/round321/gates/`)

- Lane B's L3 unit tests with the existing files: 51 passed. Non-vacuity:
  with the lateral-mixing module reverted to base, the one new test fails
  (1 failed, 50 passed); restored, 0 dirty lines.
- Receipt citation gate on this receipt: PASS, 4 citations, 0 failures,
  0 map entries failing (`gates/citation_gate_receipt.json`). The import
  shifted map entries; re-anchored (round-320 tool) and two straddling
  ranges plus the FCT-observer entry fixed by hand (`11d67e6b5`).
- Default gate (land.sh list), split across calls: citation-gate test 17,
  private-import ratchet 6, TKE terms 71, recipe 25, freshwater closure 16,
  stage face mask 2, prognostic barotropic 8, ORCA2 parallel receipt 2 =
  147 passed, 0 failed. Round-320 selection tests 3 passed.

## Resume (2026-10-20, after the dual review)

- No model value changed. Dual review: both HOLD; import inert.
- Two tests the import broke (pass at base, fail at tip) were stale
  expectations: lane B narrowed the rk3_ws GM refusal to an incomplete
  staged Treguier/nemo_iso_lap/bolus-through-fct composition, and both
  tests build exactly that (GYRE with kappa_GM = 1). The new message is
  the intended one; the match text was updated. Non-vacuity: with that
  guard disabled both fail (2 failed); restored, 0 dirty lines.
- Refusal reason corrected (Feasibility, OPEN) here and in the
  preregistration (dated correction note).
- DECISION 145 held: rn_Ud's config field has no library default
  (unset = the card must set it), and the trainable collector seeds every
  entry from the library default, so any entry needs a choice the user has
  not made (question in the final lines). Proposed bounds 0.0033-0.03 m/s:
  kappa_Redi's (330, 3000) m2/s through NEMO's aht0 = 1/2 rn_Ud rn_Ld
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:354-357`, rn_Ld
  = 200 km on ORCA2); contains NEMO's reference 0.01 and ORCA2's 0.018.
  kappa_Redi's rung-10 freeze-out waits on the same trainer entry.

## Review

- codex review: operator.

## Choices

- ASKED (brief): L3 import; D122 on seven pairs; ask if it leans away.
- UNASKED: lane B's two L3 unit-test hunks imported (tests only).
- UNASKED: one import line kept in the latlon operators that lane B
  dropped (no value change; needed by this branch).
- UNASKED (resume): two test expectations updated to the narrowed refusal
  message (tests only).
