# Preregistration — ORCA2 round 124 northwest EEN thickness association

Date: 2026-10-03. Base: `b0dfa4bdf7a746b4002506286438539028795334`.
Every hierarchy-rung-0 number is **independent** because rung 0 starts from
NEMO's own from-rest state. No configuration choice, forcing, initial state,
carried state, stabilizer, sea-ice selector, or `unmeasured_features` entry may
change.

## Admitted boundary and compiled statement

Round 123 closed northeast fraction 1's 514-cell northern `e3f_0vor` boundary
with the native row-145 F-origin permutation and left northwest fraction 3's
separate 521-cell thickness boundary first. The northeast path is an isolation
control. Both later frozen-mask boundaries remain downstream.

The executed `nn_e3f_typ=0` branch builds `e3f_0vor` from four masked T-cell
thicknesses divided by four, applies the F-grid lateral boundary, and restores
zero entries from mesh `e3f_3d`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynvor.f90:912-937`). Northwest
fraction 3 then reads the already-associated northern value at
`(ji-1,jj+1)` (`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1326-1328`).

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R124-P1 | NW fraction 3 uses the same native row-145 F-origin permutation `179-i`, sign `+1`, offset through its recorded `ji-1` operand. | Candidate northern thickness equals recorded NW `3_e3f0` bitwise at every executed cell. | Retain **REFUTED**; walk the first observed source-row/permutation alternative without changing production. |
| R124-P2 | The one-variable substitution closes exactly 521 NW thickness and denominator magnitudes and all 1,431 NW fraction magnitudes. | NW `3_e3f0`, `3_denom`, and `3_frac` each become 0/0 bit/magnitude unequal. | Stop at the first surviving NW operand; do not change production. |
| R124-P3 | NW's next source-ordered boundary is frozen `fe3mask`. | First remaining NW item is `3_mask`, with the admitted 1,154-cell census; `3_r3f` stays exact. | Name the earlier surviving item and stop there. |
| R124-P4 | NE remains bit-exact and isolated. | NE fraction 1 thickness, denominator, and quotient stay 0/0 under round 123's association. | Any NE movement refuses the instrument as confounded. |
| R124-P5 | Resolved execution scope remains literal-EEN cards only. | ORCA2, both DINO recipes, VORTEX, and VORTEX_VEC are true; GYRE and both tanks are false. | Any scope movement refuses the arm. |

## Measurement and landing bar

The measurement reuses the admitted round-120 self-describing record and the
committed round-121–123 assembly. It runs production JIT on CPU under
fp64/libm, refuses a dirty or wrongly stamped tree, and carries independent
oracle-bit, candidate-bit, wrong-row, northeast-isolation, and scope-route
plants. This round first measures only the northwest boundary. A production
association may land only if both northern paths are exact under the same
compiled statement and the complete ORCA2/GYRE/DINO/tank/card/citation gates
pass; otherwise the round is HELD.
