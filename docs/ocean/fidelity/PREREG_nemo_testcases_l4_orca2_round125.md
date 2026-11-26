# Preregistration — ORCA2 round 125 northeast EEN northern mask walk

Date: 2026-10-03. Base: `1ab56428a01900cd746ed06788c88be09f23ca7c`.
Every hierarchy-rung-0 number is **independent** because rung 0 starts from
NEMO's own from-rest state. No configuration, forcing, initial state, carried
state, stabilizer, sea-ice selector, or `unmeasured_features` entry may change.

## Admitted boundary and compiled statement

Round 124 made both northern `ff_f` and `e3f_0vor` operands bit-exact and left
northeast fraction 1's frozen `fe3mask` first: 1,160/1,160 bit/magnitude
differences, first `(j,i,k)=(147,29,0)`. Northwest's separate 1,154-cell mask
boundary is an isolation control and is not changed by this arm.

The executed source first applies the ordinary F-grid lateral boundary to
`fmask` and then freezes `fe3mask = fmask`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dommsk.f90:220-258`). The T-pivot
F-point branch with `nn_hls=2` fills the northern halo from native row 145 with
the F-origin permutation and sign `+1`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/lbcnfd.f90:722-747`). Northeast
fraction 1 consumes that mask at `(ji,jj+1)`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1323-1325`).

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R125-P1 | NE fraction 1 uses native mask row 145, F permutation `179-i`, sign `+1`. | The associated mask equals recorded NE `1_mask` bitwise at every executed cell. | Retain **REFUTED** and stop at the first surviving source-row/permutation alternative; do not change production. |
| R125-P2 | The one-variable association closes exactly 1,160 NE mask bit and magnitude differences. | NE `1_mask` becomes 0/0 bit/magnitude unequal. | Any surviving bit rejects the association. |
| R125-P3 | Because recorded northern `r3f` is already exact, NE thickness, denominator, quotient, partial, and sum remain bit-exact. | Each named descendant stays 0/0; no earlier source item moves. | Any movement refuses the arm as confounded. |
| R125-P4 | NW stays isolated at its admitted 1,154-cell mask boundary. | NW `3_mask` remains 1,154/1,154 while its thickness, denominator, quotient, and sum stay 0/0. | Any NW movement refuses the arm. |
| R125-P5 | Resolved execution scope remains literal-EEN cards only. | ORCA2, both DINO recipes, VORTEX, and VORTEX_VEC are true; GYRE and both tanks false. | Any scope movement refuses the arm. |

## Measurement and landing bar

The measurement reuses the admitted round-120 self-describing record and the
committed round-121–124 assembly. It runs production JIT on CPU under
fp64/x64/libm, refuses a dirty or wrongly stamped tree, and carries independent
oracle-bit, candidate-bit, wrong-row, northwest-isolation, and scope-route
plants. This round measures only northeast. A production association may land
only after northwest's separate mask boundary is also measured and the complete
ORCA2/GYRE/DINO/tank/card/citation gates pass; therefore a confirmed northeast
arm is **HELD** with northwest next.
