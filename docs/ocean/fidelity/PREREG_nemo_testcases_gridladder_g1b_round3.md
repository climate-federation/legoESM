# PREREGISTRATION — grid ladder rung G1b, round 3 (gridladder_r26)

Frozen before any ladder of the G1b card runs this round. Rounds 1-2's
predictions are carried unchanged.

## Record

- round22/oracle_g1b_r22 admitted by run.sh --admit-existing at 172873b01
  (twin_dir newline fix): ACQUISITION_DONE, RESTART_FULL 99 MISSING none
  (steps 2..100); the stride-10 plant misses exactly 11,21,..,91,92..99.
- The 17 new restarts (r11..r13) equal NEMO's after-step frame on ssh, u,
  v, uu_b, vv_b (0 unequal; plant: step-93 restart vs kt-92 frame, 16433
  unequal).
- Ladder root: round26/g1b_merged = r10's files + one full restart per
  step 2..100 (symlinks).

## Statement under test

`R4REF/dynvor.f90:744-748` (np_CRV) on the top F row (jj = Nje0) reads
`e1u(ji,jj+1) * pu(ji,jj+1)` from NEMO's U fold halo. Measured on NEMO's
kt-31 entry frame: halo inner column i = −u(row below the pivot, column
179 − i), i.e. the card's `perm_u` in east-face (n_lon) indexing, sign
`vector_sign_u`. Diagnosis of round 24's failed attempt (PLAUSIBLE until
the arm below runs): `fold_row` on an (n_lon+1)-column u row permutes card
columns 0..n−1, whose column 0 is the cyclic west copy, so the image is two
columns off.

## Predictions (frozen)

| id | prediction | falsifier |
|---|---|---|
| R3-V1 | GIVEN-NEMO-ENTRY rhs_terms at kt 31 with the fold-halo bracket: card vorticity term vs replay 0 unequal on the pivot U row and the V row below it (round 24: 68 + 66) | any unequal face on those two rows |
| R3-V2 | the fold halo moves nothing below the top F row: rows below the pivot unchanged vs the card without it | any change below |
| R3-L1 | GIVEN-NEMO-ENTRY restart_entry over 99 steps: first over-bar step later than 30 (round 24: 30, fold band) | first over-bar ≤ 30 |
| R3-L2 | INDEPENDENT 100 steps: first over-bar not earlier than 27 (round 24: 27 ssh / 30 u,v) | earlier than 27 |
| R3-C1 | G1b NOT certified this round (the kt-27 accumulated interior event stays) | every field at the bar under both labels over 100 steps |
