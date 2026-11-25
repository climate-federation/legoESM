# Preregistration — ORCA2 round 126 northwest EEN northern mask walk

Date: 2026-10-03. Base: `0aa51f39837929906ff6a045b32dca42cbcf2b5a`.
Every hierarchy-rung-0 number is **independent** because rung 0 starts from
NEMO's own from-rest state. The shipped rung-7 ladder is **given NEMO's
recorded entry** under Decision 52. No configuration, forcing, initial state,
carried state, stabilizer, sea-ice selector, or `unmeasured_features` entry may
change.

## Admitted boundary and compiled statement

Round 125 made northeast fraction 1's northern frozen-mask operand bit-exact
and left northwest fraction 3's separate mask first: 1,154/1,154 bit/magnitude
differences, first `(j,i,k)=(147,30,0)`. Northeast is the isolation control.

The executed source applies the ordinary F-grid lateral boundary to `fmask`
before freezing `fe3mask = fmask`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dommsk.f90:232-258`). The T-pivot
F-point branch with `nn_hls=2` fills the northern halo from native row 145 with
the F-origin permutation and sign `+1`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/lbcnfd.f90:722-747`). Northwest
fraction 3 consumes that mask at `(ji-1,jj+1)`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1326-1328`).

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R126-P1 | NW fraction 3 uses native mask row 145, F permutation `179-i`, sign `+1`, followed by its recorded `ji-1` offset. | The associated mask equals recorded NW `3_mask` bitwise at every executed cell. | Retain **REFUTED** and stop at the first surviving source-row/permutation alternative; do not change production. |
| R126-P2 | The one-variable association closes exactly 1,154 NW mask bit and magnitude differences. | NW `3_mask` becomes 0/0 bit/magnitude unequal. | Any surviving bit rejects the association. |
| R126-P3 | Because recorded northern `r3f` is already exact and zero on the differing support, NW thickness, denominator, quotient, partial, and sum remain bit-exact. | Each named descendant stays 0/0; no earlier source item moves. | Any movement refuses the arm as confounded. |
| R126-P4 | The round-125 NE arm remains exact and isolated. | NE mask and every descendant remain 0/0 after the common association. | Any NE movement refuses the arm. |
| R126-P5 | Resolved execution scope remains literal-EEN cards only. | ORCA2, both DINO recipes, VORTEX, and VORTEX_VEC are true; GYRE and both tanks false. | Any scope movement refuses the arm. |
| R126-P6 | If P1–P5 confirm, applying the same compiled association to the production frozen mask moves no ORCA2 rung-0 or rung-7 ladder row and leaves the non-executing cards unchanged. | Both 200-row ORCA2 ladders move 0 rows; GYRE residual archives/day-30 snapshot and DINO month remain byte-identical; tank/card gates do not worsen. | Any exact row leaving the bar, any earlier first debt, or any unregistered moved row holds the landing. |

## Measurement and landing bar

The measurement reuses the admitted round-120 self-describing record and the
committed round-121–125 assembly. It runs production JIT on CPU under
fp64/x64/libm, refuses a dirty or wrongly stamped tree, and carries independent
oracle-bit, candidate-bit, wrong-row, northeast-isolation, and scope-route
plants. The controlled arm changes only northwest fraction 3's frozen-mask
operand and its mathematically dependent expressions.

A production association may land only if both northern mask paths are
bit-exact, all controls fire, both ORCA2 ten-step ladders satisfy their frozen
row registries, GYRE's ladder and certified month/day-30 artifacts satisfy the
standing gate, DINO/tanks/generic cards pass, the citation plant fires, and the
required test batteries show no new failure.
