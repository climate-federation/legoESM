# ORCA2 round 76 preregistration — independent external transport walk

Date frozen: 2026-09-29

Base: `dbfc411b94d7293020792d4bbc887f796b5ff5de`

Claim label: **independent**. legoESM starts from the ORCA2 card's own ocean
state and exact kt=1 surface operands. NEMO arrays appear only in explicitly
labelled diagnostic replay and substitution arms. The production baseline
remains independent.

Sea ice remains out of scope. The card's six-entry `unmeasured_features`
tuple, selectors, 10,800 s step, thresholds, and carried state are frozen. No
model, configuration, NEMO source, or record change is authorized.

## Existing evidence, not predictions

Round 75 measured `un_adv` as the dominant non-cancelling input to the first
non-bit hybrid correction: replacing only `un_adv` reduces `zub` RMS from
`6.613384103192950e-04` to `1.1058764872516897e-06 m s-1`. With recorded
`un_adv`, the live inverse-depth/live-thickness path leaves `zFu` RMS
`1.971885366122676e-11 m3 s-1`; recorded inverse depth with live thickness
leaves `0.790927819646436 m3 s-1`.

The admitted rank-0 external-mode stream contains all 65 substeps. The
compiled ORCA2 source initializes the transport accumulator at zero, adds
`wgtbtp2 * zhU * r1_e2u` each substep, divides once by `r1_wgt2s`, then applies
the U/V boundary exchange at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:380-380,563-579,831-847`.
The step-1 transport statement consumes that result with inverse depth and
live face thickness at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:274-284`.

The search-before-build audit found and will reuse the round-74 production
runner, the round-75 substitution machinery, the admitted advective-mean
record schema and parser, the production barotropic trace, the production
metric-transport helpers, the raw-bit scorer, and the card census. The shared
parser will only be generalized with explicit expected dimensions and cycle
count; no second parser or numerical kernel will be created.

## Frozen predictions and falsifiers

1. **Record replay.** Replaying every recorded accumulator update from the
   record's own operands predicts raw-bit exact U and V accumulator exits for
   all 65 substeps, and exact pre-boundary normalized means. Any unequal replay
   cell invalidates the instrument.
2. **Production trace identity.** Enabling the private substep exposure hook
   predicts the ordinary production `un_adv`/`vn_adv` endpoint bit-for-bit.
   Any difference invalidates the exposed trace.
3. **Zero seed and weights.** The live transport accumulator predicts an exact
   zero seed and all 65 raw `wgtbtp2` values bit-exact to NEMO. Any mismatch
   stops the walk upstream.
4. **First U producer boundary.** The independent SSH mismatch is already
   present at external-loop entry, but the first U metric transport is zero
   because the initial velocity is zero. The first non-bit U accumulator
   statement is predicted at substep 2's
   `zhU * r1_e2u` metric transport, not at substep 1. A different substep or
   boundary is **REFUTED** and retained.
5. **Recorded-operand statement replay.** Each of the 65 U metric-transport
   and accumulator-add statements predicts raw-bit closure when evaluated by
   the production helpers on NEMO's own operands. Any unequal row invalidates
   the transcription.
6. **Live geometry pair.** With NEMO `un_adv` fixed, the live inverse-depth +
   live-thickness arm predicts the round-75 `zFu` RMS exactly. The recorded
   inverse-depth + live-thickness arm predicts the round-75 RMS exactly.
7. **Two-sided geometry consistency.** With NEMO `un_adv` fixed, substituting
   only NEMO thickness is predicted to worsen `zFu` relative to live/live,
   while substituting both NEMO inverse depth and NEMO thickness predicts
   raw-bit exact `zub`, corrected velocity, and `zFu`. If the thickness-only
   arm does not worsen, the two-sided consistency claim is **REFUTED**. If the
   recorded pair is not exact, the pair transcription is invalid.
8. **Disposition.** This measurement-only round predicts **HELD**. The first
   non-bit external-mode producer statement is named, but the independent SSH
   selector gap remains frozen and no landing is authorized.

Failed predictions remain **REFUTED**. External accumulation, its operands,
the downstream geometry pair, record replay, and production exposure are
separate predicates.

## Controls and validation

- A one-ULP mutation of an exact accumulator exit must fire.
- Replacing the predicted first non-bit metric transport with the live value
  must move or remove the named boundary and fire.
- Aliasing the recorded thickness arm to live thickness must fire the paired
  geometry predicate.
- A false card selector and changed round-75 baseline tuple must each fire.
- Run focused round-74 through round-76 and citation tests, both citation
  gates with a real firing plant, shared-card and tank batteries, and
  `tests/ocean/fidelity -n 12` once, one pytest battery at a time.
- Request the required separate `codex exec --sandbox read-only` review and
  retain its verdict or exact in-sandbox failure.

ASKED: walk independent `un_adv` through its compiled external-mode producer
and measure the live inverse-depth/live-thickness consistency at `zFu`.

UNASKED: model arithmetic, configuration, NEMO output, selectors, thresholds,
carried state, sea ice, and the held tracer QCO/RK statement.
