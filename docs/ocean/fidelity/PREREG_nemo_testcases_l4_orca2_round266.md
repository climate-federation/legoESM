# ORCA2 round 266 preregistration — stage-2 U RHS operator split

Date: 2026-10-13. Frozen base: `21943d712`. Scope: OMT-4, separately
labelled **independent** and **given NEMO's entry**. This round reuses round
265's matched stage-2 state and admitted records to split the completed U RHS
in NEMO's compiled order. No deck, card selector, carried state, stabiliser,
sea-ice field, or `unmeasured_features` entry may change.

## Compiled order and record contract

The resolved deck has `ln_dynadv_vec=.true.` and `nn_dynkeg=0`. NEMO writes
the stage-2 entry accumulator, then calls HPG, EEN vorticity, and vector-form
advection in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:360-421`.
The vector dispatcher calls centred KEG and then ZAD at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynadv.f90:134-139`;
centred KEG updates U at `dynkeg.f90:117-132`, and ZAD updates U at
`dynzad.f90:102-138`.

The admitted `oracle_rkstage2_terms_kt00000001.bin` contains the four
cumulative U/V boundaries at that exact program point. The existing live
operand trace exposes the already-computed production HPG, VOR, KEG and ZAD
components without adding an in-executable observer. The HPG operand and
literal streams in the same record root are available if HPG is first.

## Frozen predictions and falsifiers

1. **R266-P1 — round-265 reproduction.** Both labels retain matched stage-2
   T/S at the campaign floor and the pre-update completed U RHS substitution
   closes raw and post-associated U to the floor. Any larger residual stops
   the split as an instrument regression.
2. **R266-P2 — first accumulator.** The entry accumulator is at the floor;
   HPG is predicted to be the first cumulative boundary above `2e-10`, with
   VOR and ADV downstream. An at-floor HPG row refutes this prediction and
   advances mechanically to VOR, then ADV.
3. **R266-P3 — HPG leaf.** If HPG is first, NEMO's recorded density anomaly,
   live W thickness and live depth operands replay its literal U HPG exactly.
   The first candidate operand above the floor names the leaf statement; a
   literal replay that does not close retains HPG as a cancelling unit and
   forbids a landing. If VOR or ADV is first instead, split only that operator;
   KEG is predicted bit-exact and ZAD is the predicted first ADV half.
4. **R266-P4 — trajectory consequence.** Only a cited leaf whose source-exact
   arm closes the completed RHS reaches OMT-4 kt=1..10 and the full
   Decision-96 gates. Missing leaf operands are `ACQUISITION_NEEDED`; a
   non-closing leaf is HELD.
5. **R266-P5 — controls.** Both-label coverage, clean-tree provenance, exact
   record identity, source order, round-265 reproduction, first-boundary
   sensitivity, cumulative self-replay, leaf sensitivity, and a one-bit
   control each have a planted refusal.

## Stop and landing conditions

This is an offline, CPU-fp64/libm, matched-program-point measurement. No NEMO
run is planned. A boundary is not a leaf owner. No model statement lands unless
its NEMO inputs reproduce the oracle result and the complete private unit then
passes every Decision-96 trajectory gate. No floor, clip, damping, alternate
mask, carried-state change, configuration choice, or sea-ice change is
permitted.

ASKED choices: none. Unasked configuration choices made: none.
