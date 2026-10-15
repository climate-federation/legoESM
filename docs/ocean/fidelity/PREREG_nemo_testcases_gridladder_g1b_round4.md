# PREREGISTRATION — grid ladder rung G1b, round 4 (gridladder_r27)

Frozen before any G1b ladder or stage measurement this round.  Round 3's
card at `0afd6b2191d602691b70dd82f80d0b3c2d34aaaa` is the immutable before
arm.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round27/`.

## Record and labels

- NEMO record: round26/g1b_merged, admitted over steps 2..100; every restart
  equals its after-step frame on ssh/u/v/uu_b/vv_b.
- GIVEN-NEMO-ENTRY: entry frames kt 1..50 and restarts kt 3..100.
- INDEPENDENT: the card's own chain kt 1..100.
- Bar: max absolute error normalised by NEMO's field maximum, `1e-15`.
- Round-3 before values: GIVEN-NEMO-ENTRY has no field over the bar; its worst
  entry-frame values are ssh `1.7e-18`, u `2.6e-18`, T `0`, and its worst
  restart values are ssh `2.6e-18`, u `6.9e-18`.  INDEPENDENT first exceeds
  the bar at kt 20 (ssh), u/v at kt 36 and T at kt 35; worst ssh/u/T are
  approximately `1.5e-14` / `2.0e-15` / `2.5e-14`.

## Compiled order and candidate boundaries

The record's compiled G1b build is
`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo`.  Stage 1 accumulates HPG, VOR and
KEG in that order (`stp2d.f90:138`, `:144`, `:161`; ZAD is zero on the
one-level deck), then the vector branch writes
`(uu(Kbb) + rDt*uu(Krhs))*umask` and its V counterpart
(`stprk3_stg.f90:366-370`).  Decision 122's full ORCA2 build executes the
same statements at `stp2d.f90:141`, `:147`, `:164` and
`stprk3_stg.f90:467-471`.

Round 3 left two ordered boundaries:

1. stage-1 RHS at kt 2: 4 U / 0 V faces unequal, maximum `1.7e-24`; the
   existing observer fused HPG and KEG, so it could not name the first operand;
2. on the card's own RHS, the stage-1 velocity write: 35 U / 49 V faces
   unequal, maximum `3.5e-18`.

The HPG/KEG split will reuse the production operator-component bundle already
returned by `tendencies`; no term is re-evaluated.  Its closure is the existing
combined `KE_PGF` value bit for bit.  The velocity candidate, if reached, will
materialise only the multiply, add and mask operations written by the compiled
statement.  Both changes are G1b-card-selected; defaults and all other cards
remain unchanged.

## Frozen predictions and falsifiers

| id | prediction | falsifier |
|---|---|---|
| R4-I1 | the split observer's HPG + KEG is bit-identical to its pre-existing combined KE_PGF bucket on U and V | any unequal wet face |
| R4-I2 | a one-cell perturbation of either published half moves its named split row and makes the gate exit nonzero | plant exits zero or changes only another row |
| R4-A1 | kt-2 KEG is bit-identical to the offline replay; HPG is the first non-bit operand and carries the five-face fused-bucket residual | KEG has an unequal face earlier, or HPG is bit-identical |
| R4-A2 | the cited HPG transcription makes the kt-2 stage-1 RHS bit-identical on both faces (4/0 to 0/0) and loses no exact GIVEN row | any RHS unequal face remains or an exact GIVEN row is lost |
| R4-B1 | after A, the direct vector velocity write on the card's own RHS remains 35 U / 49 V unequal at about `3.5e-18`; source-operation materialisation makes its raw U/V outputs bit-identical | the boundary is already exact, the counts differ materially, or the candidate leaves an unequal raw face |
| R4-L1 | after each landed statement, GIVEN-NEMO-ENTRY remains at the bar over entry frames 1..50 and restarts 3..100; unequal-face counts and maxima fall at the named statement | any field leaves the bar, or the named statement does not strictly improve |
| R4-L2 | after each landed statement, INDEPENDENT remains not certified; first-over-bar and worst values may move by last-place lottery and will be registered without treating a later first step alone as progress | every field is at the bar over kt 1..100 (G1b certified) |

If R4-A1 is refuted, the first unequal half named by the split owns candidate
A and the receipt keeps the failed prediction.  If A changes the next boundary,
B is re-measured before any B edit.  Each landing is followed by a complete
rescore of both labels before the next statement is changed.

## Landing and stopping rules

A statement lands only when its own unequal-face count becomes zero given
NEMO's entry operands, no exact GIVEN row is lost, and its G1b-only selection
is mechanically pinned.  A later INDEPENDENT first-over-bar step is not by
itself improvement; the statement's unequal-face count and maximum carry the
claim.  G1b is CERTIFIED only if every scored field is at the bar under both
labels through kt 100.  A boundary needing a new NEMO record stops the round;
no `makenemo` or `mpirun` is run here.

## Choices

ASKED: split HPG from KEG; land the cited first non-bit statement; then inspect
and, if confirmed, land the cited stage-1 vector velocity write; select both
only on G1b; rescore both labels after each landing.  No configuration value,
default, timestep, record, score definition or carried-state choice is made.
