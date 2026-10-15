# PREREGISTRATION — grid ladder rung G1b, round 2 (gridladder_r24)

Frozen before any ladder of the G1b card runs on NEMO's G1b record.
Round 1's predictions (PREREG_nemo_testcases_gridladder_g1b_round1.md,
section 5) are carried UNCHANGED; this file adds only what the admitted
record now makes testable.

## Record (admitted this round, before any ladder)

- Reference: round22/oracle_g1b_r22/ref_100step, 100 steps, no stp_ctl stop.
- Twins r01..r10, 100 steps each; step frames r01..r09 byte-identical to
  r10's at kt = 1..100.
- G1's check_records.py on twin a (r10) with r01 and the reference: PASS;
  its seven plants each fire.
- Restarts holding a record: 82 of steps 2..100 (missing 11, 21, .., 91 and
  92..99, see the round-2 receipt); each equals NEMO's after-step frame on
  ssh, u, v, uu_b, vv_b (0 unequal).
- Ladder root: round24/g1b_merged = twin a's files plus the 72 full
  restarts of r01..r09 (symlinks).

## Labels

- INDEPENDENT: the G1b card's own chain from its own initial state,
  kt = 1..100, scored on NEMO's after-step frames.
- GIVEN-NEMO-ENTRY: kt = 1..50 from NEMO's entry frames (given_entry arm);
  kt = 2..100 from NEMO's restart of kt - 1 wherever the record holds one
  (restart_entry arm). Never mixed (D52).

## Predictions (frozen)

| id | prediction | falsifier |
|---|---|---|
| R1-F-P1..P4 | as in round 1, section 5 (first non-bit row at kt = 1; the velocity-form external update; then sum / stage update / np_CRV; no relative vorticity or KEG before kt 1 stage 2) | as in round 1 |
| R2-G1 | geometry identity on the G1b root equals G1's (every mesh array the card carries bit-identical) | any geometry row unequal |
| R2-L1 | INDEPENDENT: a field leaves the 1e-15 bar before kt = 100 (G1b NOT certified on the current card) | no row over the bar in 100 steps |
| R2-L2 | the first over-bar region is the open interior, not the fold, seam or south wall (the fold machinery is G1's, certified) | the first over-bar row is a fold, seam or south row |
| R2-S1 | S-P2 of round 1: max abs U and V of NEMO's record stay below 0.05 m/s | any step above 0.05 m/s |
