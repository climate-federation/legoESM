# ORCA2 round 268 preregistration — G1 fold statements transfer

Date: 2026-10-13. Base: `ec14bd3d857ec0158cc949ac57ccf0efb6900e2b`.
This file is frozen before either G1 statement is selected on an ORCA2 or OMT
card and before any round-268 numerical measurement.

## Authorized change

Transfer, without re-derivation, the two selector-gated statements landed on
the geometry-ladder branch at `43f4f070c18da9d5f39b4e1dcb1cd0284fc6369a`:

1. NEMO's EEN vorticity trend reads the V transport on the fold-image row
   above the pivot T row (`dynvor.f90:784-800` in the compiled ORCA2 tree;
   G1 round-2 commit `6613a0982`).
2. NEMO's split-explicit solver evaluates the V-face live depth on that same
   wet fold-image row (`dynspg_ts.f90:419-421`, consumed at `:437`, with the
   same form at `:480`; G1 round-3 commit `697f82b39`, review fix
   `43f4f070c`).

Both selectors remain default-off. Select both explicitly on the OMT cards and
the shipped ORCA2 ocean card; do not change any other card, selector, default,
deck, initial state, carried state, stabiliser, sea-ice declaration, or
`unmeasured_features` entry.

## Frozen predictions and falsifiers

- R268-P1 — transfer fidelity. The imported implementation and its default-off
  controls agree with the two G1 receipts; removing either selected option
  makes its direct plant fail. Any semantic drift from the pinned G1 tip
  refutes the transfer.
- R268-P2 — matched stage-2 owner. With the fold-image EEN statement selected,
  the cumulative VOR row at the round-267 matched program point closes under
  both **independent** and **given NEMO's entry** labels: maximum error at or
  below `2e-10 m/s2`. A larger error under either label refutes ownership and
  returns to the round-267 leaf split.
- R268-P3 — completed RHS. After R268-P2, the completed stage-2 RHS closes to
  the same `2e-10 m/s2` floor under both labels. If VOR closes but the completed
  RHS does not, the transfer lands only if its own full gates pass and the next
  operator boundary is named; no downstream value is attributed to VOR.
- R268-P4 — atomic unit. With both transferred statements and the existing
  complete private fold/transport unit, OMT-4 completes kt=1..10 under both
  labels without the live-thickness refusal. Any earlier refusal freezes the
  landing and names its first passive boundary.
- R268-P5 — Decision-96 landing predicate. No exact certified row leaves the
  bar; the first-over-bar row is toward NEMO or unchanged; among RMS-changing
  rows, toward exceeds away; the rung-0 sea-surface maximum is not worse.
  Failure of any predicate holds the candidate.
- R268-P6 — shared-card isolation. GYRE's certified ladder/year, DINO's month,
  and tank/generic-card gates are byte-identical or remain inside their frozen
  bars because none selects these two options. Any shared-card move is a red
  gate, not a tolerated effect.
- R268-P7 — controls. Missing selector coverage, altered source order,
  incomplete record identity, non-passive tracing, a moved certified row, and
  a one-bit synthetic violation each make the round gate refuse.

## Measurement order

1. Transfer the two pinned implementations and select them on every ORCA2/OMT
   card that executes the statements.
2. Re-run round 267's matched stage-2 source-order measurement under both
   labels; score VOR before the completed RHS.
3. If the operator rows close, integrate the existing complete private unit
   through OMT-4 kt=1..10 under both labels.
4. Only if the finite prefix completes, run OMT-1..4, rung-0, rung-10, GYRE,
   DINO, tanks, moved-row census, focused tests, citation gate and its plant,
   and the required fidelity battery.

No NEMO acquisition is requested: the admitted OMT-4 frames and stage-term
records contain the frozen oracle side needed for this transfer measurement.
