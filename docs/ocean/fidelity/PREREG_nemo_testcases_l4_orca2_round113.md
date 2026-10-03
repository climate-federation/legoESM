# Preregistration — ORCA2 round 113 southern EEN thickness association

Date: 2026-10-02. Base: `877e334f10676745e6600a295d92f38a191bf472`.
Scope is one hierarchy-rung-0 source statement, measured **given NEMO's
recorded entry**. No configuration, forcing, initial state, carried state,
threshold, stabilizer, sea-ice selector, or `unmeasured_features` entry may
change.

## Executed oracle path

Round 112 closed the southern `ff_f` numerator association and left the first
non-bit source-ordered item at the southern `e3f_0vor` operand: seven cells,
first `(j,i,k)=(0,28,0)`. The denominator differs at the same seven cells;
the separately open frozen-mask operand differs at 66 cells.

The compiled rung-0 source builds `e3f_0vor`, calls the generic F-grid lateral
boundary operation, and only then replaces its zero values with `e3f_3d`
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynvor.f90:912-937`). The lateral
boundary call supplies no `kfillmode`; on the executed double-precision path a
closed southern halo therefore takes the default constant zero
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:1811-1820,1864-1872`).
The replacement operand `e3f_3d` was itself read as an F-grid field with
`jpfillcopy` (`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/domzgr.f90:179-188`),
whose southern association copies the nearest interior row
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:2021-2046`). Thus the
executed southern `e3f_0vor` operand is the source row of the carried mesh
`e3f_0`, not a cyclic read of the northern fold row.

The candidate will replace only that southern `e3f_0vor` association in the
literal EEN quotient. The still-open southern `r3f` and `fe3mask` associations
remain unchanged.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R113-P1 | The unmodified replay reproduces round 112: `south_e3f0` is first, with exactly 7 bit and magnitude differences, first `(0,28,0)`. | Exact census and first index match the admitted round-110 record. | Any mismatch means the baseline moved; stop and reconcile before testing a candidate. |
| R113-P2 | NEMO's executed southern `e3f_0vor` is the copy-filled carried mesh `e3f_0`: row 0 repeats source row 0 and row `j>0` reads row `j-1`. | That one-variable association makes `south_e3f0` bit-exact on every executed cell. | Any remaining `south_e3f0` bit rejects the association and forbids a landing. |
| R113-P3 | The same substitution makes `south_denom` bit-exact because its seven differences are co-located with `south_e3f0`; the next raw source-ordered item is `south_mask`, with 66 unequal cells and first `(0,29,0)`. | `south_e3f0` and `south_denom` each have zero unequal bits; `south_r3f` is exact; `south_mask` retains exactly 66 unequal bits. | A different first item owns the continued walk; preserve this prediction as REFUTED. |
| R113-P4 | The statement executes on the same literal-EEN cards registered in round 112: ORCA2, VORTEX, VORTEX_VEC, and the two NEMO-faithful DINO recipes; GYRE and the flux-form tanks do not execute it. | Resolved-card routing equals this set and its planted mutation fires; every executing card's binding gate is green. | Any additional card executes: add its binding gate before landing. |
| R113-P5 | No certified ORCA2 row leaves AT-BAR and no first-over-bar row moves earlier. | Complete rung-0 and rung-7 200-row comparators pass; every moved row is registered; shared-card gates pass. | Any red landing predicate holds the change and names the exact row. |

## Measurement and landing bar

Extend the round-112 fraction walk and its admitted self-describing round-110
record. Compare baseline and the single southern-thickness candidate on
executed levels and rank-complete owned cells under production JIT on CPU with
fp64/x64/libm. Oracle-bit, candidate-bit, and routing plants must fire. A direct
unit control must fail when cyclic association is restored.

The statement may land only if it is bit-exact given NEMO's recorded operands
on every executing card, both ORCA2 ladder gates satisfy the frozen predicate,
and the required GYRE/DINO/tank/generic-card/citation/push gates pass.
Otherwise the round is HELD at the first red line. No downstream mask,
fraction, product, accumulator, fold, or later-substep item is in scope.
