# ORCA2 round 294 preregistration — raw-hv substep continuation

Date: 2026-10-16

Round 293 proved, on the independent OMT-4 kt=1 stage-1 external solve,
that replacing only the reconstructed fold-image `hv_0` by NEMO's stored
operand closes the first substep-1 V-face-depth boundary from 13 unequal
active faces (maximum 719 m) to zero.  The immediate `zhV` consumer was a
null control because midpoint V is zero.  This round carries that one
recorded operand through the passive external-substep trace and names the
next valued statement in compiled order.  It changes no production package,
card, configuration, carried state, or NEMO record.

The executing record build extrapolates midpoint V and SSH at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:460-499`, forms
`zhvp2_e` from the stored `hv_0` at `:509-520`, forms `zhV` at `:530-536`,
updates SSH at `:546-558`, accumulates the external transport at `:563-580`,
and then evaluates the pressure, Coriolis, drag, and velocity update through
`:601-739`.  The source registry follows those statements for every one of
the 65 external substeps.

Every scientific number is labelled **independent OMT-4, kt=1 stage 1**.
The arm replaces only the compact V reference-depth slot; its U slot is the
unchanged reconstructed production value.  The traced and untraced arm runs
must end array-identically before any intermediate is read.

## Frozen predictions and falsifiers

1. **R294-P1 — admission, precision, passivity.**  The inherited round-222
   substep/operand twins and round-280 stage record admit; the resolved card
   remains vector-form with the literal fold unit and completed-stage top-V
   exchange off; CPU production JIT and fp64/libm print.  The raw-`hv_0`
   traced and untraced completed states are array-identical.  Any miss
   refuses scientific output.
2. **R294-P2 — one-variable arm.**  The override's U reference depth is
   bit-identical to production, while its V reference depth differs only on
   round 293's 13 fold-band cells and equals the card's stored NEMO `hv_0`.
   Any U movement, off-fold V movement, or census change refuses the arm.
3. **R294-P3 — inherited closure.**  With only raw `hv_0`, substep 1's
   midpoint V depth is bit-exact against NEMO and every earlier source row
   remains at the campaign floor.  A surviving substep-1 depth difference
   refutes the round-293 causal result and refuses further attribution.
4. **R294-P4 — next boundary.**  The next valued row is substep 3's midpoint
   V depth, fold-local on 35 faces, because the compact `ssvmask` then makes
   the free-surface term live and exposes the T-pivot northern neighbour.
   An earlier/later row, different operand, or different census is retained
   as **REFUTED** and becomes the measured first boundary.
5. **R294-P5 — registered pair.**  If R294-P4 confirms, the already-cited
   raw-V-surface-mask plus T-pivot-neighbour pair from round 245 is the
   cancelling candidate; this round names it but does not partially land it.
   If the boundary differs, the pair is not credited.
6. **R294-P6 — non-vacuity.**  Passivity, source order, arm isolation,
   inherited closure, first-boundary, locality, and one-bit plants must each
   refuse.  A green plant invalidates the measurement.

The outcome is measurement-only and **HELD** unless the complete private
fold unit is subsequently shown to pass its full Decision-96 gates; that
trajectory score is not silently inferred from a substep closure.  Decision
125's completed-stage top-V exchange and the resolved stage mask remain off.
Decision 126 remains pending and the shipped runoff setting is unchanged.
Sea ice and the shipped card's `unmeasured_features` tuple are unchanged.

Independent review: operator, as required for this lane.
