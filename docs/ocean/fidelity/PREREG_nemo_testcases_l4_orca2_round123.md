# Preregistration — ORCA2 round 123 northeast EEN thickness association

Date: 2026-10-03. Base: `2c977174f180b85470f81317e59e275ea5819f4c`.
Every hierarchy-rung-0 number is **independent** because rung 0 starts from
NEMO's own from-rest state. No configuration choice, forcing, initial state,
carried state, stabilizer, sea-ice selector, or `unmeasured_features` entry may
change.

## Admitted boundary and compiled statement

Round 122 made northeast fraction 1's northern `ff_f` bit-exact and left its
next source-ordered operand, `e3f_0vor`, unequal in 514 executed cells. The
northwest path's separate 521-cell thickness boundary and both later mask
boundaries remain downstream and are not scored as this round's claim.

The executed `nn_e3f_typ=0` branch builds `e3f_0vor` from the four masked T
thicknesses divided by four, applies its F-grid lateral boundary condition,
and only then restores fully dry values from mesh `e3f_3d`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynvor.f90:912-937`). The NE
fraction then reads the already-associated value at `(ji,jj+1)` before its
current-row operands
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1323-1325`).
Therefore the controlled arm changes only NE fraction 1's northern thickness
selection from cyclic row 0 to the existing, source-built northern row.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R123-P1 | NE fraction 1 must hold the already-folded northern `e3f_0vor` row rather than cyclically wrapping to row 0. | The candidate northern thickness equals the recorded NE `1_e3f0` operand bitwise at every executed cell. | Preserve **REFUTED**; score the first observed alternative (preceding-row F permutation or mesh fill) without changing production. |
| R123-P2 | This one-variable substitution closes exactly 514 NE thickness and denominator magnitudes. | NE `1_e3f0` and `1_denom` each move from 514/514 to 0/0 bit/magnitude unequal. | Stop at the first surviving NE operand; do not change production. |
| R123-P3 | NE's next source-ordered boundary is its frozen `fe3mask`. | The first remaining NE item is `1_mask`, with the admitted 1,160-cell census; `1_r3f` stays exact. | Name the earlier surviving item and stop there. |
| R123-P4 | The northwest path is outside this controlled arm. | NW fraction 3 retains its admitted 521-cell `e3f_0vor` boundary byte-for-byte. | Any NW movement refuses the instrument as confounded. |
| R123-P5 | Resolved execution scope remains literal EEN cards only. | ORCA2, both DINO recipes, VORTEX, and VORTEX_VEC are true; GYRE and both tanks are false. | Any scope movement refuses the arm. |

## Measurement and landing bar

The measurement reuses the admitted round-120 self-describing record and the
committed round-121/122 assembly. It runs production JIT on CPU under
fp64/libm, refuses a dirty or wrongly stamped tree, and carries independent
oracle-bit, candidate-bit, cyclic-wrap, northwest-isolation, and scope-route
plants. This is a measurement-only northeast walk: no `packages/` or card file
may change. A production statement is considered only after the separate
northwest 521-cell boundary has been walked in a later round.
