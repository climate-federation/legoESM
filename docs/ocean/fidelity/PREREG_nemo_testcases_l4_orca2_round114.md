# Preregistration — ORCA2 round 114 southern frozen-mask association

Date: 2026-10-03. Base: `faf8b047431b6ece8441dedc8a7aa632b3a60ff8`.
Scope is one hierarchy-rung-0 source statement, measured **given NEMO's
recorded entry**. No configuration, forcing, initial state, carried state,
threshold, stabilizer, sea-ice selector, or `unmeasured_features` entry may
change.

## Executed oracle path

Round 113 closed the southern `e3f_0vor` and denominator association and left
the first non-bit source-ordered item at the southern frozen mask: 66 cells,
first `(j,i,k)=(0,29,0)`.

The compiled rung-0 source applies the ordinary F-grid boundary operation to
`fmask` and only afterwards freezes `fe3mask = fmask`
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dommsk.f90:229-258`).  That call
does not pass `kfillmode` or `pfillval`.  On the executed double-precision
boundary path the closed southern halo therefore takes the default constant
zero (`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:1811-1820,
1864-1872,1999-2005`).  The literal EEN quotient then reads this frozen mask
at `jj-1` (`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1244-1248`).

The candidate will replace only the southern `fe3mask` association in the
literal EEN quotient.  Its `ff_f`, `e3f_0vor`, and `r3f` associations remain
as landed in rounds 112-113.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R114-P1 | The unmodified replay reproduces round 113: `south_mask` is first, with exactly 66 bit and magnitude differences, first `(0,29,0)`. | Exact census and first index match the admitted round-110 record. | Any mismatch means the baseline moved; stop and reconcile before testing a candidate. |
| R114-P2 | NEMO's executed southern frozen mask is zero-filled at global `j=0` and reads row `j-1` above it. | That one-variable association makes `south_mask` bit-exact on every executed cell. | Any remaining mask bit rejects the association and forbids a landing. |
| R114-P3 | With the mask exact, `south_denom`, `frac_south`, `sum_west_center`, and `sum_all` are all bit-exact; no recorded fraction item remains non-bit. | Every item in the round-110 self-describing fraction record scores zero unequal bits. | The first remaining item owns the continued walk; retain this prediction as REFUTED. |
| R114-P4 | The statement executes on the registered literal-EEN cards: ORCA2, VORTEX, VORTEX_VEC, and the two NEMO-faithful DINO recipes; GYRE and the flux-form tanks do not execute it. | Resolved-card routing equals this set and its planted mutation fires; every executing card's binding gate is green. | Any additional card executes: add its binding gate before landing. |
| R114-P5 | No certified ORCA2 row leaves AT-BAR and no first-over-bar row moves earlier. | Complete rung-0 and rung-7 200-row comparators pass; every moved row is registered; shared-card gates pass. | Any red landing predicate holds the change and names the exact row. |

## Measurement and landing bar

Extend the round-113 fraction walk and its admitted self-describing round-110
record. Compare baseline and the single southern-mask candidate on executed
levels and rank-complete owned cells under production JIT on CPU with
fp64/x64/libm. Oracle-bit, candidate-bit, and routing plants must fire. A
direct unit control must fail when cyclic association is restored.

The statement may land only if it is bit-exact given NEMO's recorded operands
on every executing card, both ORCA2 ladder gates satisfy the frozen predicate,
and the required GYRE/DINO/tank/generic-card/citation/push gates pass.
Otherwise the round is HELD at the first red line. No downstream product,
accumulator, fold, or later-substep item is in scope.
