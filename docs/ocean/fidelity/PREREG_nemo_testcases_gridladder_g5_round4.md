# Preregistration — grid ladder G5, round 4 (gridladder_r34), independent-chain drift

**Frozen:** 2026-10-16, at `e965b1b7f`, before any measurement this round.
Record: `gridladder_rounds/round31/oracle_g5_r31` (admitted round 2).
Bar: the ladder's existing normalized 1e-15; no threshold change.

## Already on file (round 3, `gridladder_rounds/round33`)

- kt-1 staged walk: stage 2 T is unequal in 1 cell (1.1e-16) and stage 2 S in
  2 cells (7.1e-15); stages 1 and 3 are bit-exact on all levels.
- The step-final S row is unequal from kt 2 (2 cells) under BOTH labels; the
  independent S count grows 2,2,3,4,5,5 (kt 2..7); given-entry stays 1-3.
- Independent ssh/U/V first unequal at kt 7 (ssh 428 cells).

## Predictions

1. **First leave at all.** The independent chain first leaves NEMO at kt 1
   stage 2 (T 1 cell, S 2 cells), NOT at kt 7 ssh; the first step-final leave
   is kt 2 S. kt 7 ssh is the first leave of a DYNAMICS field. Falsifier: any
   stage-1 or step-final field unequal before kt 1 stage 2, or kt 1 stage 2
   bit-exact on re-measurement.
2. **Carried state at kt 7 entry.** Diffing every leaf the card carries
   between steps (after kt 6) against what a NEMO restart hands it: the only
   unequal prognostic is S (≤ 5 cells, last bit); ssh, uu_b, vv_b, u, v, T and
   the six barotropic histories are bit-equal. Falsifier: any other carried
   leaf unequal in a field NEMO's restart also holds.
3. **Causal swap at kt 7.** Running kt 7 from the card's carried state with
   ONLY NEMO's S swapped in makes kt 7 ssh/u/v bit-exact; running it from
   NEMO's restart with only the card's carried S swapped in reproduces the
   independent kt 7 ssh leave. Falsifier: either arm disagrees → the owner is a
   carried leaf the restart does not hold (to be named from P2's diff).
4. **Owner statement.** The kt 1 stage-2 T/S leave is a tracer statement of
   stage 2 (`stp_RK3_stg`, stage 2: `tra_sbc_RK3`/runoff/`tra_qsr`/`tra_zdf`)
   reading a Kmm thickness the card forms differently, the stage-2 analogue of
   G4 round 9's stage-3 stored-ratio fix. Walked in compiled order; the first
   non-bit statement is named with its `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1`
   citation and its `ORCA2_OMIP_L4` (D122) line at the enclosing SUBROUTINE.
   PLAUSIBLE only; falsified if the leave is in a non-tracer statement.
5. **Certification.** If a cited stage-2 statement lands (G5-selected) and
   makes kt 1 stage 2 exact, both labels are bit-exact or AT-BAR on every
   surface and all-level row kt 1..10 → G5 CERTIFIED. Otherwise HELD with the
   owner named.

## Also this round (not predictions)

- One fast traced-`jit` synthetic test per top-drag consumer that fails when
  the consumer is removed (review caveat: the only failing test is slow).
- Restore `g5_record_not_admitted` and `nemo_post_si3_surface_records` in
  `ORCA2G5_UNMEASURED` if they still describe unmeasured rows.
