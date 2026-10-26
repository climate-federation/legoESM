# ORCA2 round 64 preregistration — vector-invariant ZAD walk

Date frozen: 2026-09-28

Base: `1026f84027c730f4f61b70863e69c38c3ba9d9a3`

Claim label: **given NEMO's recorded operands**.  Independent-start and
month-scale claims are out of scope for this statement walk.  Sea ice, its six
selectors, and the ORCA2 card's `unmeasured_features` tuple remain frozen.

## Source-first scope

The operator completed the repaired two-rank acquisition at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round63/acquisition_vector_split/orca1ice_vector_advection_split_np2`.
Its own admission reports `PASS`, 21 self-describing fields per rank, physical
EOF, and byte-identical ocean and ice restart calibration.  This round first
re-runs that admission and every plant from the committed launcher.

The ORCA2 card executes vector-invariant momentum advection, not flux-form
UP3.  Its compiled dispatcher calls KEG then ZAD.  Round 63 discharged KEG at
zero unequal active owned faces.  The executing ZAD branch has
`ln_vortex_force=.FALSE.` and `ln_zad_Aimp=.FALSE.` and performs, in order:
area-times-`ww` products, U/V two-point sums, vertical velocity differences,
transport-times-difference products, the carried-interface sum, live-QCO
thickness scaling, and the Krhs subtraction at
`ORCA2_ORCA1ICE_OMIP_L4_R63VADVSP/BLD/ppsrc/nemo/dynzad.f90:102-137`.

## Frozen predictions and falsifiers

1. **Admission.** The fresh record remains admissible at the round-63 producer
   commit.  CONFIRM only if both rank files retain the declared 21-field order,
   physical EOF, zero effective Stokes drift, distinct global origins, and all
   four restart digests.  Any schema, payload, stamp, or restart refusal stops
   scientific scoring.
2. **Literal ZAD replay.** A statement-ordered fp64 replay predicts `0` unequal
   active owned U faces and `0` unequal active owned V faces against the
   recorded after-ZAD accumulator.  Any unequal bit REFUTES the prediction and
   names the first differing compiled substatement by replaying the ordered
   intermediates; it is not rounded into an AT-BAR claim.
3. **Controls.** A one-ULP mutation of one active U replay value must produce
   one refusal per rank.  Replacing live `e3u_Kmm` by a reference-thickness
   field is unavailable in this record and therefore is not used as a control;
   no carried operand may be reconstructed.
4. **First-non-bit disposition.** If the literal ZAD replay is exact, both
   children of the admitted after-VOR→after-ADV boundary are discharged and
   the earlier framing that this boundary itself named a model defect is
   RETRACTED: it measured NEMO's nonzero physical update, not a legoESM–NEMO
   mismatch.  The walk then returns to the already-open QCO mixed boundary.  If
   ZAD is non-bit, its first unequal substatement owns the next round.
5. **Landing.** No model statement is preregistered to land.  This round changes
   no file under `packages/`; ORCA2, GYRE, DINO, and tank trajectories therefore
   cannot move.  A physics change discovered post-measurement is held for a
   separately preregistered landing round under the complete shared-card gate.

Failed predictions remain **REFUTED** in the receipt.  The gate reports exact
cell counts separately from numerical maxima, stamps fp64 inputs, parses the
self-describing stream rather than a predicted byte count, and carries a
synthetic violation that proves the exact comparison fires.

## Required validation and OPEN

Run the resolved-card execution census before scoring ZAD, the record admission
and all five acquisition plants, the ordinary ZAD gate and its one-ULP plant,
focused tests, the card battery, the single permitted `tests/ocean/fidelity
-n 12` battery, citation gate plus shifted-line plant, and a separate
`codex exec --sandbox read-only` review.  Because no model file is expected to
change, GYRE trajectory gates are not applicable unless that expectation is
refuted.

ASKED: admit and walk ORCA2's compiled stage-2 ZAD statement.

UNASKED: configuration, selector, threshold, carried state, forcing,
stabiliser, sea ice, and NEMO arithmetic changes.
