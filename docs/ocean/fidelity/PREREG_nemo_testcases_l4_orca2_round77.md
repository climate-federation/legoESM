# ORCA2 round 77 preregistration — substep-2 external U-transport pair

Date frozen: 2026-09-29

Base: `a84ac5ec2a416aefdec450d746c83b2a32aa8888`

Claim label: **independent**. legoESM starts from the ORCA2 card's own ocean
state and exact kt=1 surface operands. NEMO arrays appear only in the three
explicitly labelled diagnostic substitution arms. The production baseline
remains independent.

Sea ice remains out of scope. The ORCA2 card's six-entry
`unmeasured_features` tuple, selectors, 10,800 s step, thresholds, and carried
state are frozen. No model, configuration, NEMO source, or record change is
authorized.

## Existing evidence, not predictions

Round 76 measured the first non-bit external-mode U producer at substep 2:
`zhU = e2u * ua_e * zhup2_e`, compiled at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:530-536`.
The statement differs on 8,568 / 8,568 active rank-0 columns, with maximum
`23.712296310346574 m3 s-1` and RMS `2.6045999142426335 m3 s-1`. Both operands
are already non-bit: `ua_e` differs on 8,568 / 8,568 with maximum
`1.3951550259711822e-06 m s-1`, while `zhup2_e` differs on 8,568 / 8,568 with
maximum `0.015548358737760282 m`.

The compiled source forms `ua_e` from the three external velocity history
levels at `dynspg_ts.f90:460-474`, forms `zsshp2_e` and then `zhup2_e` at
`dynspg_ts.f90:476-521`, and multiplies the two operands at
`dynspg_ts.f90:530-536`. Round 18 already supplies a controlled substitution
template for this same multiplication given NEMO's recorded entry; round 77
reuses the round-76 independent production trace, admitted advective-mean
record, production metric-transport helper, raw-bit scorer, and card census.
No second numerical kernel or parser will be created.

## Frozen predictions and falsifiers

1. **Round-76 baseline.** The inherited round-76 classifier predicts
   `PASS_EXTERNAL_TRANSPORT_WALK`, the same substep-2 boundary, and the exact
   frozen baseline count, maximum, and RMS. Any changed value invalidates the
   instrument.
2. **Arm isolation.** Four arms differ only in the source of `ua_e` and
   `zhup2_e`: live/live, NEMO/live, live/NEMO, and NEMO/NEMO. A source ledger
   mismatch invalidates the arm.
3. **Pair closure.** The NEMO/NEMO arm predicts raw-bit exact `zhU` on all
   8,568 active columns. Any unequal cell invalidates the statement replay or
   record alignment.
4. **Dominant owner.** The NEMO-`ua_e`/live-`zhup2_e` arm predicts a smaller
   `zhU` RMS than both the independent baseline and the live-`ua_e`/NEMO-
   `zhup2_e` arm. Failure of either strict inequality **REFUTES** `ua_e` as the
   dominant non-cancelling operand and is retained.
5. **Depth-only direction.** The NEMO-`zhup2_e`-only arm predicts it will not
   close `zhU` raw-bit exactly. Exact closure **REFUTES** the predicted
   `ua_e` owner and assigns the walk to the independent sea-surface-height
   path, which remains frozen at `STOP_SELECTOR_GAP`.
6. **Disposition.** This measurement-only round predicts **HELD**. If `ua_e`
   owns the smaller residual, the next walk is its compiled substep-1 update;
   if `zhup2_e` owns or closes the residual, work stops at the frozen
   independent SSH selector gap. No production landing is authorized.

Failed predictions remain **REFUTED**. Operand magnitude, isolated arm
direction, pair closure, and downstream ownership are separate predicates.

## Controls and validation

- A one-ULP mutation of the exact NEMO/NEMO product must fire.
- Aliasing the NEMO-`ua_e` arm to the live/live source ledger must fire.
- Swapping the two single-arm RMS values must fire the owner predicate.
- A changed round-76 baseline and false card scope must each fire.
- Run the focused round-75 through round-77 and citation tests, both citation
  gates with a real firing plant, and the permitted wide fidelity battery once
  only if this round adds or changes a test outside the focused set.
- Request the required separate `codex exec --sandbox read-only` review and
  retain its verdict or exact in-sandbox failure.

ASKED: measure the one-variable `ua_e` and `zhup2_e` substitutions and their
pair at independent ORCA2's first non-bit substep-2 `zhU` statement.

UNASKED: model arithmetic, configuration, NEMO output, selectors, thresholds,
carried state, sea ice, the registered downstream geometry pair, and the held
tracer QCO/RK statement.
