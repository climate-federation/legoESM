# NEMO testcase L2 GYRE phase 3 — round 82 drag-boundary receipt

## Outcome

**HELD.** The acquired Round-81 external-step record was admitted and the
source-order walk succeeded. It found two real compiled-program boundaries:
materialization of the tracer-cell bottom-drag coefficient before its face
average, followed by NEMO's distinct window-entry inverse-depth expression.
The paired candidate made every recorded external-step operand bit-exact
through the combined U/V trends; the first non-bit input became `slow_u`, as
Round 81 predicted.

The candidate nevertheless fails Rule 12. Against both the immutable
Decision-36 before arm and the sealed Round-79 histories arm, kt2 T, S, U, V,
and SSH do not move at all. In particular the required kt2 U/V residuals stay
`2.7478404751243857e-12` and `3.305560306813421e-12`. The first-over-bar row
stays kt2 U/V, but later rows worsen by as much as 7,815 row-scale float64 ULPs,
far beyond the two-ULP allowance. The production candidates were therefore
reverted in commit `71a7cdb93dff03c202b458f4ee691167c2cb54c2`; the final tree
retains the measurement seam and fail-closed walk only.

No 30-day, LOCK_EXCHANGE, or OVERFLOW candidate run was started after this
decisive GYRE rejection. They are not needed to reject an already ineligible
candidate, and no result is inferred for them.

## Preregistration and provenance

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round82.md`, committed as
`07e7c681f7c7aeb0548c343871f7e4994b874c7e` before the acquired record was
scored. Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round82/`.

The record producer is
`295a42edc9d9f45707a5349097e7a0183f57463c`; its stream is 10,439,164 bytes.
The normal admission retained exactly 46/70 byte-identical inherited records,
24 classified changed records, and 264 admitted changed fields. All six record
plants (stamp, header, truncation, replay ULP, swap ULP, U-midpoint ULP) and
both twin-admission plants (consumed field, unexpected inventory) exited
nonzero. The ordinary arithmetic replay and the Round-77 U-midpoint identity
were bit-exact.

The first clean walk, at `f472a236bd6a6d87225d46e986b20a6e8693904f`,
**REFUTED** the frozen slow-forcing prediction: the first non-bit statement was
substep-1 `drag_coefficient_u`, 124/580 wet faces, absolute maximum
`6.776263578034403e-21`. The diagnostic at
`512ad153a470f970f4bd573de14bae5daa6182d5` replayed both face averages from
the live tracer-cell coefficient and matched the oracle on 580/580 U and
570/570 V faces. This identifies a missing stored-array rounding boundary,
not a different drag law or face stencil.

The materialization candidate `e43daf15901dae819e0233bdeadf6e2dfcb734ca`
moved the first non-bit row to substep-1 `inverse_depth_u` (324/580 wet faces,
absolute maximum `5.421010862427522e-20`). The entry-inverse candidate at
`22c45f6f5a934c074f40cb671f8fc35228011612`, and the final instrument-only
commit `090b9319c3d65cc57cc67baefd0a79f331c064dc`, both **CONFIRMED** the frozen
prediction: the first non-bit statement was substep-1 `slow_u`, 580/580 wet
faces, absolute maximum `1.0529650291768787e-11`. Every preceding registered
row, including both drag coefficients, both inverse depths, and both combined
trends, was bit-exact. The history-ULP, slow-U-ULP, and null-slow-U controls all
printed `FIRED` and exited 1.

Two pre-result instrument attempts are retained as instrument failures: the
first rejected a named staggered shape mismatch, and the next rejected an
unexpected stacked diagnostic rank. Neither emitted a scientific JSON result.
The preregistered average-before-square hypothesis was also **REFUTED** by its
non-vacuity check: power-of-two scaling made that rewrite algebraically and
bitwise equivalent, so it was removed before a candidate measurement.

## Compiled-source findings

This section and every citation below bind to the acquired card's own compiled
branch under `cfgs/GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo`.

The active standard-drag loop builds `rCdU_bot` from the Kmm bottom velocity
and stores the square-root result at the tracer point
(`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/zdfdrg.f90:229-235`). A separate
bottom-only loop later consumes that stored array to form U/V face
coefficients with two-point averages
(`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1497-1500`). The
Round-82 live-cell replay being exactly equal while the fused JAX face result
was not is direct evidence that this intervening binary64 array boundary owns
the 124/580 and 149/570 coefficient differences.

At external-window entry NEMO separately constructs `hu_e/hv_e` and
`hur_e/hvr_e`; the inverse is reference reciprocal divided by `1+r3`, in both
the forward Kmm and centred Kbb arms
(`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:363-375`). The
explicit non-wetting/drying branch consumes the frozen face coefficient,
entry velocity, and that inverse depth in written multiplication order
(`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:703-706`).

The next owner is therefore upstream input, not another external-step formula.
The compiled forcing producer depth-averages the 3-D RHS
(`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/stp2d.f90:202-207`), then applies
the baroclinic drag correction
(`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/stp2d.f90:225-227`), and finally
adds the wind term before `dyn_spg`
(`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/stp2d.f90:229-235`). Round 83
must walk those already recorded operands in exactly that order.

## Rule-12 table

| Lane | Registered movement and result |
|---|---|
| GYRE kt2 external substeps | **DIRECT MATCH / CANDIDATE HELD.** Candidate first mismatch moved from `drag_coefficient_u` to the imported `slow_u`; every formula row through `trd_u/trd_v` became bit-exact. |
| GYRE kt=1..10 | **FAIL.** 70 trajectory rows compared against both recorded arms. kt1 stays exact; kt2 and kt3 have zero movement in every field; first-over-bar remains kt2 U/V; no status changes; maximum worsening is 7,815 row-scale ULPs. |
| Every moved GYRE row | Exactly these 27 rows moved: kt5 `before.u`; kt6 `before.u`; kt7 `after.uu_b`, `after.vv_b`, `before.u`, `before.v`; and, for each of kt8, kt9, kt10, `after.uu_b`, `after.vv_b`, `before.S`, `before.T`, `before.ssh`, `before.u`, `before.v`. The full per-cell improved/worsened census is sealed in `round82_gyre_vs_decision36.json` and `round82_gyre_vs_round79.json`. |
| GYRE days 1..30 | **NOT RUN — candidate already rejected.** The immutable before arm remains `decision36_nemo_face_shear/after_day_gap.json`; no long-horizon claim is made. |
| LOCK_EXCHANGE | **NOT RUN — candidate already rejected.** The shared drag statements would execute only where its recorded `rCdU_bot` is nonzero; no neutrality claim is made. |
| OVERFLOW | **NOT RUN — candidate already rejected.** No neutrality claim is made. |
| DINO | **SHARED-STATEMENT RISK.** DINO's leap-frog card carries its own histories, but it calls the same drag-law/face-coefficient program. A materialization boundary could therefore move DINO bits even though the Decision-37 absolute-history continuation does not execute there. No numerical claim is made. |
| ORCA2 | **UNMEASURED WITH SPEC.** Input: independently produced ORCA2 NEMO restart and legoESM NEMO-identity state at the same instantaneous Kbb frame. Fields: T, S, U, V, SSH, tracer-cell and face drag coefficients, entry inverse depths, and six absolute barotropic histories. Masks: native NEMO tmask/umask/vmask staggering. Statistic: elementwise fp64 bit equality plus normalized L-infinity per row for kt=1..10. Bar: 1e-15. Falsifier: any AT-BAR loss, earlier first-over-bar, or wet-point history/drag/inverse mismatch. |

Because the GYRE candidate failed, no carried-state or restart format changed in
this round. The prior Decision-37 restart compatibility and loud-failure tests
remain untouched.

## Review and tests

The mandated separate review command was attempted after the candidate and
Rule-12 table were known. It failed before model startup. Its terminal line,
quoted verbatim, is:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Per the operator's 2026-09-13 08:30 instruction: **independent review
unavailable in-sandbox**; work continued. No reviewer verdict is claimed, and
the physics is held independently by the mechanical Rule-12 failure.

Focused CPU/fp64 checks on the final trace-only tree: the Round-82 walk,
repository citation-gate, complete `test_zdf_dynzdf_composition.py`, and NEMO
literal barotropic-continuity suites passed, 43 tests total in 30.91 seconds.
The known unrelated `test_rk3_ws_differs_from_rk3_and_is_finite` failure was
not encountered. The clean-tree citation gate passed all 7/7 compiled-source
citations with zero map-audit failures. Shifting the compiled standard-drag
citation by two lines exited 1 with `SYMBOL-NOT-AT-LINE`.

## Choices, uncertainty, and retractions

Choices made: none. Decision 37 was already answered YES, but its pair remains
conditional on kt2 U/V movement and no worsening; this round mechanically
failed both requirements. No bar, selector, carried-state convention, parent
arm, or configuration was changed.

Retracted in the tool and implementation, not only in prose:

- the frozen claim that slow forcing was the first mismatch on the unmodified
  program was REFUTED and remains printed as REFUTED by the first artifact;
- the average-before-square explanation was REFUTED and removed;
- the materialization and entry-inverse production candidates were reverted
  after the ladder rejection; only trace output can still expose their inputs;
- no Round-79 day improvement is promoted into a Round-82 claim.

Uncertainty: the walk proves the next recorded external-step boundary is
`slow_u`; it does not yet identify which `stp2d` producer input first differs.
The missing independent reviewer is procedural uncertainty, not permission to
weaken the mechanical HOLD.

## OPEN — exact handoff to round 83

1. Reuse the admitted kt2 K(r)hs/stage/slow-forcing records; no NEMO acquisition
   is requested. Preregister and walk the compiled `stp2d` producer in this
   exact order: 3-D RHS, reference-thickness depth average, baroclinic drag
   correction, wind, final `Ue_rhs/Ve_rhs`. The first non-bit input owns the
   next walk.
2. Keep the Round-82 materialization and entry-inverse commits held. They may
   be reconsidered only as a preregistered cancelling pair with the newly
   identified upstream forcing owner, and only if the immutable Decision-36
   ladder moves kt2 U/V toward the bar with no >2-ULP worsening.
3. If a future pair passes kt1..10, then and only then run days 1..30 plus the
   recorded LOCK_EXCHANGE and OVERFLOW rows, state DINO shared-statement risk,
   and retain the ORCA2 unmeasured-with-spec contract above.
4. Retry the exact adversarial Codex review outside this sandbox and quote its
   verdict before any physics candidate ships.

ACQUISITION_NEEDED: NONE

DECISION_NEEDED: NONE
