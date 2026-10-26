# NEMO testcase Lane 4 — ORCA2 card round 29 preregistration

Date: 2026-09-26

Parent: `23eb11d18b9a081b1acb764b8ee52f1c3f265ebf`

Status: **PREREGISTERED BEFORE ROUND-29 SCIENTIFIC SCORING.**

Round 29 executes round 28's first OPEN item.  It records the exact inputs and
output of each production EEN call during the kt=1 stage-2 exposure, first on
the parent and then on round 28's consumer-local raw-F arm.  The comparison is
**independent with Decision-52 SSH**.  It does not mix a given-NEMO-entry
operator result into the trajectory result.

The six sea-ice selectors and the ORCA2 card's `unmeasured_features` tuple
remain frozen.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round29/`.

## Compiled statements and production boundary

The admitted build forms EEN's reciprocal live F thickness at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:733-738`,
forms the absolute/relative vorticity numerator at `:741-780`, forms the U/V
face transports at `:782-786`, and assembles/adds the tendency at `:788-803`.
The frozen reference F thickness used by the reciprocal is built and exchanged
at `:907-937`.  The admitted `ocean.output` resolves `ln_dynvor_een = T` and
`ln_dynvor_msk = F`.

The production implementation passes those same three operand classes to
`pv_flux_al81_partial_cell`: total vertex vorticity, live vertex thickness,
and the two masked face transports.  Round 28's EEN-only source diff changed
only the live vertex-thickness override at that call boundary.  Round 29 adds
a WRITE-only observer around the existing production function; it does not
rederive the operator.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R29-P1 | Round 28's first active EEN movement is already a denominator-only experiment. | At the earliest call whose EEN output changes, total-vorticity numerator, U/V velocities, U/V face thicknesses, and U/V masks are bit-identical; only live vertex thickness differs. | Any numerator or transport input differs at that earliest changed call, or live vertex thickness does not differ. |
| R29-P2 | The denominator-only output reproduces round 28's exposed stage-2 EEN result. | Parent and raw-F output digests equal round 28's respective stage-2 digests and the raw arm changes the published 413,554 U / 412,558 V cells. | Either digest or either unequal-cell count differs. |
| R29-P3 | The observer is passive. | The parent exposed stage-2 U/V arrays are bit-identical to round 27/28's saved parent arrays. | Either array changes. |
| R29-P4 | The outcome gate binds. | A one-representable-value plant in a captured transport input is refused. | The planted transport change passes. |

Failed predictions remain **REFUTED**.  If P1 confirms, round 28's requested
denominator/numerator/transport split is discharged by measurement and the
next walk splits the denominator's compiled sub-operands: frozen `e3f_0vor`,
live `r3f`, and `fe3mask`.  If P1 refutes, the first differing non-denominator
operand owns the next walk.

## Landing and stop rules

- Commit this preregistration before adding or running the observer.
- Reuse the admitted ORCA2 record, fp64/libm precision policy, CPU backend,
  production JIT, surface forcing, Decision-52 entry, and stage-2 exposure.
- Reuse round 28's exact consumer-local source arm; do not create a new scheme,
  configuration field, state field, or numerical method.
- Revert every experimental `packages/` edit before the receipt.
- No Decision-54 statement lands from this attribution round.
- Keep the raw-`e3w` refusal unchanged.  No stabilizer, clipping, NEMO-source
  edit, sea-ice edit, scoring change, or acquisition is authorized.

## Choices

ASKED: Decision 54 and round 28's OPEN item authorize this compiled-order EEN
operand census.

UNASKED: none.
