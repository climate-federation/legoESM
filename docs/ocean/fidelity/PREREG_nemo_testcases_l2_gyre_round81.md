# Preregistration: NEMO-testcases L2 GYRE round 81 kt=2 external-step record

Date: 2026-09-13. Frozen after reading the Round 79/80 receipts, the completed
Round 80 parent comparison and adversarial review, the admitted Round 77 U
midpoint record, and the running Round 77 target's compiled source, but before
rerunning the live U walk, inspecting any new kt=2 external-step record, or
changing a production numerical statement.

## Inherited retraction and magnitude boundary

Round 80's completed comparison is authoritative. The exact parent at
`0078f9cc851c921176afd8e30660373129321712` and the Round 79 candidate at
`add5cbd555b99ebe32ce4e46ae69538f67951883` have zero movement in all 954
registered rows. In particular, `GYRE-zco.kt2.before.u` and
`GYRE-zco.kt2.before.v` remain respectively
`2.7478404751243857e-12` and `3.305560306813421e-12` from NEMO. Round 79's
prediction that these rows would move is therefore **REFUTED**; its receipt's
later reinterpretation that kt=3 was the first possible movement is withdrawn.
The absolute-history candidate remains **HOLD** under Decision 37, although it
stays in the one shared program while the compensating/next owner is walked.

The separate review's exact verdict was `HOLD`. It required the failed kt=2
prediction to remain refuted and required direct V, SSH, and six-history
coverage because the admitted Round 77 record exposes only U. Pair enforcement
and format-3 migration had no separate code defect. This round dispositions
those findings literally; no review finding is silently averaged away.

## Compiled source order

The running target is
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90`. Its compiled
branch copies the whole-step two-dimensional forcing into `ssh_frc`, `zu_frc`,
and `zv_frc`, then subtracts the Kmm barotropic Coriolis contribution from the
two momentum forcings at `:288-324`. It initializes the current U/V/SSH values
from Kmm while preserving the six absolute b/bb histories on continuation at
`:339-378`.

Within each external substep, the compiled branch forms U and V midpoints from
the absolute current/b/bb triplets at `:481-505` and forms the SSH midpoint at
`:512-518`. It then forms metric transports and the continuity update at
`:549-583`, the backward SSH midpoint and pressure-gradient terms at `:627-642`,
the ENE Coriolis and explicit bottom-drag trend at `:644-666`, and the active
vector-form U/V update from entry, pressure, combined trend, and slow forcing
at `:678-702`. The resolved GYRE card has `ln_dynadv_vec=T` and `ln_wd_dl=F`, so
the later flux-form and implicit-wetting/drying arms do not execute. Boundary
exchange precedes the absolute history swap at `:744-795`.

The corresponding two-dimensional source computes the vector-form forcing by
depth-averaging the complete three-dimensional RHS, adding baroclinic drag and
wind, and then calls the split-explicit program at
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/stp2d.f90:127-214,220-311`.
These are the statements the new record observes; no inferred discretisation
or stabilizer is substituted.

## Frozen record and measurement

First rerun the existing Round 79 direct gate at the clean preregistration
commit. Prediction: all substep-1 U coefficient/current/history/midpoint rows
remain bit-exact, and the first U non-bit row remains substep-2 `un_e`, 580/580
wet faces, maximum `3.032539284029834e-09`. The existing history-ULP plant must
exit nonzero. Any changed count, magnitude, earlier row, stale producer/stamp,
non-fp64 array, disabled JIT, or green plant falsifies this prediction.

The missing oracle acquisition is one new WRITE-only kt=2 stream from a new
target derived file-by-file from the admitted Round 77 source card. For every
one of 50 substeps it records, in compiled order and over the exact owned
`ntsi:ntei,ntsj:ntej` extent:

1. substep number and the three forward midpoint coefficients;
2. current/b/bb/mid U, V, and SSH;
3. midpoint U/V depths, metric transports, SSH forcing, divergence, and the
   continuity result;
4. the four backward coefficients and backward SSH midpoint;
5. U/V pressure gradients;
6. U/V Coriolis-only trends;
7. U/V drag coefficients and inverse depths, then combined Coriolis-plus-drag
   trends;
8. U/V slow forcing and the updated U/V result after boundary exchange; and
9. the U/V/SSH current values immediately after the absolute-history swap.

The stream header records version, kt, 50-cycle count, full dimensions, fp64,
and owned bounds. The reader consumes every byte, requires finite values,
requires substeps 1 through 50 in order, requires the exact field count and
size, and checks the direct swap identities. Header, truncation, swapped-field,
replay-ULP, stamp, inherited consumed-field, and unexpected-inventory plants
must all exit nonzero. The source and target namelists, final restart, and mesh
must be byte-identical; every inherited changed record must remain explicitly
classified by the shared admission gate.

Prediction for the new record: the already observed U substep-1 rows reproduce
the admitted Round 77 U record bit-for-bit. With the absolute six-history
representation in place, all substep-1 U/V/SSH current, b, bb, and midpoint
rows are predicted bit-exact against the live shared-program trace. The first
non-bit live row is predicted to be `slow_u` or `slow_v`, because those are the
first recorded operands imported from the already non-bit three-dimensional
kt=2 state after midpoint, continuity, pressure, Coriolis, and drag have used
the predicted-exact external state. This ownership prediction is falsified by
any earlier V/SSH/history/continuity/pressure/trend mismatch, by both slow
forcing rows being bit-exact, or by a different first source-order row. A
failed prediction remains **REFUTED** in the next receipt; no post-hoc row
reordering is allowed.

No model comparison is made from the new record in this round because the
operator alone may build and run NEMO. The next round must first admit the
record and then compare the production-JIT trace in the frozen source order.
If the first non-bit row is an input, its compiled producer owns the next walk.
If all inputs are exact and a shared result is not, only that first statement
is eligible for a separately preregistered numerical candidate.

## Rule 12 and exclusions

No production numerical statement is changed in this acquisition round.

| lane | frozen disposition |
|---|---|
| GYRE source order | Rerun the admitted U gate; acquire V/SSH/six-history and full kt=2 substep-1 pressure/forcing/update coverage; do not infer an unseen row |
| GYRE kt=1--10 | Preserve Round 80's exact parent comparison: all 954 rows unchanged and the kt=2 U/V movement prediction REFUTED |
| GYRE days 1--30 | Preserve Round 79's controlled recorded-before result as prior evidence only; no new candidate or scratch toggle |
| LOCK_EXCHANGE-zco | Preserve Round 79's 50-row result; no statement changes in this round |
| OVERFLOW-zps | Preserve compiled non-execution of the cross-window continuation branch; no statement changes in this round |
| DINO | SHARED-STATEMENT RISK: DINO uses its separate leapfrog carry, while the shared restart surface remains exposed; its 96--98% per-row cancellation forbids a neutrality inference |
| ORCA2 | UNMEASURED-WITH-SPEC: independently aligned T/S/U/V/SSH plus six absolute histories, native staggered masks, elementwise fp64 equality and normalized L-infinity through kt=1--10; reject AT-BAR loss, earlier first-over-bar, or any wet-face history mismatch |

No configuration/default, carried-state representation, selector, threshold,
stabilizer, year harness, reconciliation gate, freshwater pair, #1484 guard,
held manifest, canonical NEMO source, or existing evidence is changed. The new
target and record names are acquisition provenance, not scientific choices.
