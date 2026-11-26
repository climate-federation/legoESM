# Preregistration — ORCA2 round 135 GYRE merge and passive FCT walk

Date: 2026-10-04. Base before merge: `a5b9627e5`. Incoming GYRE tip:
`a3be519e072d67491d4c47e0853a9325db331e61`. Merge commit:
`785712759baf1a5da6f3362779d4df45087b39dc`. Ladder results are **given
NEMO's recorded entry**; the month and step-36 walk are **independent** from
the rung-0 card's own climatological T/S, zero velocity, and zero sea surface.

The incoming source statements are the ORCA2 compiled vector depth average at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:194-200` and the carried external-mode
Coriolis operand and subtraction at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:320-324`. No deck, forcing,
configuration, sea-ice selector, carried-state definition, stabilizer, metric,
threshold, or `unmeasured_features` entry changes.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R135-P1 | The merge is complete and its sole conflict is the citation-map key for the shared barotropic call site. | Both exact parents are ancestors; no unmerged path; default citation audit has zero unmapped citations. | Stop and report the exact unresolved or dropped hunk. |
| R135-P2 | Both 200-row ORCA2 ladders move because ORCA2 executes the two incoming statements over partial cells; no exact row leaves exactness. | At least one row moves on each ladder and exact-row losses are zero. | Mark **REFUTED** and retain the full row register. |
| R135-P3 | The independent rung-0 trajectory remains finite through step 35 and first becomes non-finite at step 36, T `[86,159,0]`. | The month gate repeats that exact boundary. | Mark **REFUTED** and use the observed first step/field/cell as the only next boundary. |
| R135-P4 | A private post-step FCT-input exposure can reproduce the unobserved step bit-for-bit because it substitutes diagnostics only after the ordinary stage program completes and uses no callback in the traced computation. | Every non-diagnostic returned field and the ordinary replay are bit-identical, including NaN payloads; a passivity plant fires. | Reject the instrument and every internal FCT number. |
| R135-P5 | With the exact implicit vertical transport exposed, the first active-support non-finite FCT row is the pre-limiter antidiffusive flux, as round 134 predicted. | Every source-earlier row is finite and at least one antidiffusive-flux value is non-finite. | Mark **REFUTED**, name the earliest observed source-ordered row, and do not skip forward. |
| R135-P6 | The target cell `[86,159,0]` has the same first non-finite internal row as the global active-support census. | The two first-row names agree. | Mark **REFUTED** and retain both boundaries. |
| R135-P7 | Instrumentation changes no public selector or card and the shared GYRE ladder/year gate remains within its registered round-217 predicates. | Private-hook scope tests pass; GYRE first-over-bar is not earlier and year movement is within the standing floor allowances. | No instrument landing; report the first red gate row. |

## Round bar

Register every moved ladder row after the merge, remeasure the independent
month's first non-finite boundary, and admit an FCT internal number only after
the observed step is bit-identical with and without the private exposure. Then
walk the compiled FCT order: two-step upstream predictor, low-order flux,
antidiffusive flux, limiter, corrected divergence, and final RHS. The round
lands only if the merge and private seam pass the shared GYRE and card gates;
otherwise it is HELD with the first exact red row.
