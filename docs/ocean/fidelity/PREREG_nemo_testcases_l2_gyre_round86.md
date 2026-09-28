# Preregistration: NEMO-testcases L2 GYRE round 86 stage-1 ZAD operand walk

Date: 2026-09-13. Frozen at legoESM `de9c585fb45e` after reading the Round
84--85 receipts, the admitted Round-64 record contract, the Round-41/46
literal WZV/KEG/ZAD replays, and the executing compiled source, but before
running a Round-86 scientific comparison or changing production numerics.
Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round86/`.

## Inherited boundary and magnitude rank

Round 85 landed the Decision-38 histories/drag/ZAD-operand bundle. Its current
production before arm is `round85/bundle_kt1_10.json` and
`round85/bundle_day_gap.json`; the Decision-36 artifacts remain immutable
historical controls. At the landed tip kt2 U/V maxima are
`2.7377110452773967e-12` / `3.284922138989399e-12`, kt3 T/S are
`1.627497246303733e-04` / `6.327735185607253e-06`, and day-30 T RMS is
`1.2397011295506804e-02 K`.

Round 84 left two distinct boundaries. LDF is the first non-bit cumulative
statement at `2.5292467120726215e-14` U / `3.502735092670824e-14` V, while
the ZAD addition first carries the approximately `1.9e-9` magnitude. The
source-order reconstruction also missed the independently exposed live total
by `8.470329472543003e-22`. This round follows the magnitude first, while
retaining both smaller debts explicitly.

## Executing compiled statements

The admitted record was produced by the executing GYRE branch. It constructs
stage-1 `ww` from Kbb U/V and the Kaa free-surface level, then calls KEG and
ZAD (`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:155-175`). The
executing QCO W routine vertically integrates horizontal divergence and the
`r1_Dt*e3t_3d*(r3t(Kaa)-r3t(Kbb))` stretching term from the bottom upward
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/sshwzv.f90:285-310`). ZAD then
forms `e1e2t*ww`, adjacent-face sums, Kmm velocity differences, the live
face-thickness divisor, the carried `zWdz` value, and the interior/bottom
accumulator updates in written order
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynzad.f90:102-138`).

These are live branches: the Round-64 header resolves stage 1, Kmm=Kbb,
64-bit `wp`, vector C2 advection, QCO, `ln_vortex_force=.false.`, and records
the post-ZAD accumulator immediately after that call. No generic source or
dead arm is used.

## Reused instrument and ordered rows

Extend the existing Round-84 reader and current production-JIT/fp64/libm
WRITE-only operand trace. Do not add another binary parser. Admission must
retain the Round-64 producer SHA, complete schema/EOF checks, its exact
43-record/20-classified/132-difference inventory, and zero owned-and-defined
difference projection for the consumed kt2 stage-1 record.

Score the following native wet rows in compiled consumption order:

1. Kmm U and V;
2. `ww` (including only defined/consumed W levels and owned stencil cells);
3. `e1e2t*ww`, adjacent U/V face sums, and Kmm vertical velocity differences;
4. live U/V face thickness, r3 factor, reciprocal face area, and masks;
5. carried `zWdzU/zWdzV` before each level update;
6. every interior update, bottom update, and final post-ZAD accumulator;
7. reconstructed source-order final versus the live total, plus the live
   kernel's own routine-boundary association.

Every row reports dtype, element count, unequal-cell count, absolute maximum,
reference maximum, and normalized maximum. A literal numpy replay of the
compiled ZAD loop on NEMO's own recorded inputs must remain zero unequal before
any legoESM ownership claim. Undefined W halos are excluded only by the
already-admitted owned/defined contract, never by a new post-hoc mask.

## Frozen prediction and falsifiers

The prediction is that Kmm U/V remain bit-exact, while `ww` is the first
magnitude-bearing non-bit operand and carries at least 90 percent of the
post-ZAD absolute maximum on both faces when all later ZAD arithmetic is
replayed literally. Live face thickness/r3 may be non-bit, but its one-variable
substitution is predicted below one percent of the final post-ZAD maximum.

The prediction is **CONFIRMED** only if record admission and literal
given-input replay are exact, the first non-bit operand is `ww`, the W-only
replay contribution reaches the 90-percent criterion on both faces, all
earlier rows are bit-exact, and final source-order/live-total closure becomes
bit-exact after using the same effective stage-1 operands. It is **REFUTED** if
an earlier operand differs, W owns less than 90 percent on either face,
thickness/r3 reaches one percent, the literal replay is non-exact, or closure
remains non-bit. A refutation remains in the receipt; the measured first
non-bit and first magnitude-bearing statements become the next walk.

Three maximum-residual one-ULP plants must each print `PLANT_FIRED` and exit
nonzero: an owned/consumed W value, a live face-thickness value, and the final
accumulator association. Each plant must change its named row's exact count or
maximum; perturbing a dry, undefined, zero, or already-irrelevant cell fails.

## Conditional implementation and Rule 12

No production change is pre-authorized. If the first magnitude-bearing
statement has all inputs bit-exact and only its result non-bit, the smallest
literal compiled-source transcription may be tested in the one shared
NEMO-identity implementation. An imported non-bit operand, failed literal
replay, or failed closure lands only the diagnostic and hands the exact
upstream statement forward.

| lane | frozen Round-86 disposition |
|---|---|
| GYRE kt2 ZAD chain | Admit Round 64; score every listed operand/intermediate/update in compiled order; require given-input exactness and exact effective-RHS closure |
| GYRE kt=1..10 | If production changes, compare against `round85/bundle_kt1_10.json`; preserve every AT-BAR row, forbid earlier first-over-bar, and register every moved row |
| GYRE days 1..30 | If and only if the ladder passes, run the fixed 30-day member and compare against `round85/bundle_day_gap.json` |
| LOCK_EXCHANGE-zco | If a shared statement changes, measure its recorded kt1..10 rows against the Round-85 post-history arm |
| OVERFLOW-zps | Prove the changed statement does not execute or measure its recorded kt1..10 rows; no neutrality inference |
| DINO | **SHARED-STATEMENT RISK:** DINO uses the shared W/ZAD and LDF implementations on its leapfrog path; its 96--98% regional cancellation forbids neutrality inference |
| ORCA2 | **UNMEASURED-WITH-SPEC:** align Kbb/Kaa SSH, Kmm U/V, W divergence/stretching inputs, live face thickness/r3, every ZAD carry/update, cumulative RHS, and T/S/U/V/SSH on native masks through kt1..10 in fp64; reject any wet-input mismatch, AT-BAR loss, or earlier first-over-bar |

No configuration/default, selector, coefficient, timestep, stabilizer,
carried-state representation, restart format, year harness, reconciliation
gate, freshwater pair, #1484 guard, held manifest, NEMO source, or NEMO
executable changes. Decisions 37--38 are not reinterpreted.
