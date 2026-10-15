# ORCA2 round 284 — round-282 review correction and top-F-row hold

Date: 2026-10-15

Status: **LANDED** for the round-282 review corrections and the default-off
top-F-row implementation; **HELD** for selecting that implementation on any
ORCA2 card.  The frozen round-276 Decision-96 classifier is restored, the
landed area statement is cited to the EEN routine ORCA2 executes, its omitted
stage-2 falsifier is now measured, and the selector tests fail closed.  The
separate top-F statement loses the D96 RMS vote on all seven measured
card/label combinations, so `een_nemo_top_f_u_halo` remains false in
production.

Sea ice, runoff, NEMO decks, NEMO sources, and records did not change.
Decision 126 remains pending and is not re-raised.

## Provenance

- Lane base: `c48413be5`.
- Frozen preregistration: `PREREG_nemo_testcases_l4_orca2_round284.md`, commit
  `e45b2d525`.
- Review correction and default-off top-F implementation: `950817a64`.
- Current measurement producer: `ff3ed57f1212dac4eeb1767a91a9e3ab701ae8f4`.
- Fail-closed plant repair: `47ee34d24`.
- Evidence root:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round284`.
- `stage_classification.json` SHA256:
  `f9c1ce647fc93debd9ea5c1ca583f914b3e5d2e87835eb4616634800a547f2c5`.
- `top_f_decision.json` SHA256:
  `1570c02eabe644a6cb901e02e30bc94cbbfff9a3d3d3870ee220a375deabf1af`.

## Compiled statements and executed branch

The shipped ORCA2 deck selects EEN.  The executing `vor_een` np_CRV branch
forms the relative circulation and applies `r1_e1e2f` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:762-766`; the F-cell reciprocal is
is built from `e1f*e2f` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/domhgr.f90:155-156`.  The previous
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:648-651` anchor is inside dead
`vor_ens` for this deck and is retracted.  The citation gate now checks the
enclosing routine, and a planted
`vor_ens` expectation at the live span refuses with
`ROUTINE-CONTEXT-MISMATCH`.

At the top F row, the same live np_CRV expression reads its northern
`e1u*pu` operand at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:762-766`.
The imported grid-ladder routing supplies NEMO's U-fold halo behind the new
default-false selector.  The e1u fold-halo replay hunk from `78d2eb350` is
dead on this lane and was not presented as a production statement.

## Independent-review dispositions

1. **Classifier restoration — CONFIRMED.**  Reclassifying frozen round-282
   artifacts through the unmodified round-276 predicate leaves A eligible on
   all seven card/label combinations.  B and A+B each refuse on OMT-1,
   OMT-2, and OMT-3 with `first non-bit checkpoint moved`.  The summary SHA256
   is `8ee80fe430885cbcd70120c8a0ae718070115ec99719461b2ac25069535c6b98`.
2. **Citation correction — CONFIRMED.**  Both the preregistration and receipt
   now cite the live EEN branch.  The context plant fires even though another
   np_CRV text match exists in the file.
3. **Missing R282-P2 falsifier — CONFIRMED NOW; prior process claim
   RETRACTED.**  Under both `independent` and `given NEMO's entry`, the
   `[19,33,3]` after-VOR absolute error is
   `1.2863307536456779e-08 -> 3.3881317890172014e-21` m/s2.  A planted equal
   error refuses.
4. **Selector binding — CONFIRMED.**  Tests fail if the call site hard-wires
   `een_nemo_zeta=False`, require west-cyclic and south-row behavior, refuse
   missing raw operands, and raise when the area arm is paired with non-NEMO
   metric weighting.
5. **Omitted top-F statement — MEASURED, HELD.**  Its local fold-band maximum
   moves slightly toward NEMO (`1.178054441167015e-06 ->
   1.1780244133151932e-06` m/s2), while fold-band RMS moves away and every
   trajectory loses the D96 majority vote.  A local improvement is not a
   landing argument.

The exact-loss control initially raised the correct underlying exception but
escaped the round-284 CLI rather than printing a fail-closed status.  That
failed control artifact is retained.  Commit `47ee34d24` catches the existing
Decision-96 gate exception; the fixed plant exits 2 with
`previously exact rows left the bar: [(1, 'entry', 'T')]`.

## Top-F Decision-96 census

Labels are kept separate.  Every first-over-bar row is unchanged and no exact
row is lost.

| Card | Label | toward | away | equal | D96 |
|---|---|---:|---:|---:|---|
| rung 0 | independent | 89 | 96 | 0 | HOLD |
| OMT-1 | independent | 27 | 114 | 0 | HOLD |
| OMT-2 | independent | 39 | 142 | 0 | HOLD |
| OMT-3 | independent | 40 | 141 | 0 | HOLD |
| OMT-4 | independent | 68 | 117 | 0 | HOLD |
| rung 10 | given NEMO's entry | 70 | 115 | 0 | HOLD |
| rung 10 | independent | 70 | 115 | 0 | HOLD |

Aggregate, for inventory only: 403 toward / 840 away / 0 score-equal.  The
aggregate gate status is `FAIL_R282_D96_DECISION`; each individual card is
already a sufficient refusal.  The statement stays default-off until G1b
certification or a later user-approved predicate applies.

## Shared gates and tests

- The current-tip GYRE ten-step residual archive is byte-identical to round
  282, SHA256
  `3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.
  All 30 daily snapshots are byte-identical; day 30 SHA256 is
  `3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`.
- DINO/card/tank focused gates pass 61/61 in 306.75 s.  The citation/recipe/
  freshwater push battery passes 128/128 in 408.81 s.  Round-specific tests
  pass 15/15 in 43.90 s.
- The single `tests/ocean/fidelity -n 12` battery reports 3,580 passed, seven
  skipped, and 13 failures.  Five G1 failures carry LLVM `Cannot allocate
  memory` / worker-crash evidence and all five exact IDs pass in fresh serial
  `-p no:xdist` processes.  The remaining eight are the already registered
  GYRE round-129, two GYRE round-33, OMT-1 module-edge, round-228 citation,
  dirty-escape scope, worktree-stamp, and SI3 scalar-math provenance reds.
- Default and round-282 citation gates pass; the rigid-shift plant and new
  enclosing-routine context plant both fire.

Independent review: **operator**.  Per the current lane instruction, no
nested review was attempted in this sandbox.

## Configuration choices

- **ASKED:** none.  Decision 126 remains pending and is not re-raised.
- **UNASKED:** none.
- Sea ice remains exactly as the ORCA2 card carries it, including its
  `unmeasured_features` tuple.

## OPEN

1. Resume round 283's matched stage-1 vector-V split from the resolved 3-D
   mask statement before re-testing held fold unit B.
2. Keep top-F selector C off on ORCA2.  Reconsider it only after G1b is
   certified or under an explicit later decision.
3. Leave shipped runoff unchanged until Decision 126 is answered.
