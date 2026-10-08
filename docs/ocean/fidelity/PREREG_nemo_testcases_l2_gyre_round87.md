# Preregistration: NEMO-testcases L2 GYRE round 87 stage-1 WZV input walk

Date: 2026-09-13. Frozen at legoESM `40506ad55c22` after reading the Round
85--86 receipts, the Round-64 admitted record, and the executing compiled
source, but before running a Round-87 scientific comparison or changing
production numerics. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round87/`.

## Inherited boundary and magnitude rank

Round 85 is the current landed before arm:
`round85/bundle_kt1_10.json` and `round85/bundle_day_gap.json`. At that arm,
kt2 U/V maxima are `2.7377110452773967e-12` /
`3.284922138989399e-12`, kt3 T/S are `1.627497246303733e-04` /
`6.327735185607253e-06`, and day-30 T RMS is
`1.2397011295506804e-02 K`.

Round 86 proved that W alone reproduces all of the approximately `1.9e-9`
stage-1 ZAD boundary on both faces. This is larger than the retained first
cumulative LDF non-bit boundary (`2.5292467120726215e-14` U and
`3.502735092670824e-14` V), so this round follows W by magnitude. The separate
source-order/live-total association debt (`8.470329472543003e-22` on both
faces) remains registered and may not be hidden by the W result.

## Executing compiled statements

The GYRE driver calls `stp_2D` with `Kbb=Kmm=Nbb` before any RK stage
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:187-201`). Within
`stp_2D`, the executing vector-invariant branch computes `r3t(Kaa)` directly
from the already-stored `ssh(Kaa)`, calls WZV with Kbb velocity, and then calls
ZAD; the external-mode solve occurs only later
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:141-175`, `:300-311`).
The WZV call's `np_velocity` branch forms each U/V metric-thickness-velocity
product, differences the face fluxes, divides by live T thickness, and then
materializes `e3t*hdiv`
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/divhor.f90:123-154`). Its active
QCO recurrence forms `r3t(Kaa)-r3t(Kbb)`, multiplies by `r1_Dt*e3t_3d`, adds
the materialized divergence, masks, and carries bottom-up
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/sshwzv.f90:293-300`).

The stage-1 transport correction is a different later statement: only after
the external solve does `stp_RK3_stg` form `zub/zvb` and `zFu/zFv`; for the
executing vector-invariant stage 1 it explicitly does not call WZV again
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:274-339`). A
post-solve transport-form W must therefore be scored separately from the
pre-solve velocity-form W that ZAD consumes.

## Instrument and ordered rows

Reuse the Round-46 fail-closed stage reader and Round-83 Round-64 admission.
Admission must retain producer `3b3b045bd9e03b60330204e7590e4c4470b7a0ca`,
43 byte-identical records, 20 classified changed records, 132 admitted
differences, and zero owned-and-defined differences for the kt2 stage-1
record. The inherited literal NumPy WZV replay on NEMO's inputs must remain
zero unequal.

Score the live production operands against the NEMO record in compiled order:

1. Kbb U/V;
2. U/V metric, live face thickness, metric-thickness product, velocity
   product, west/south neighbour, each directional difference, and their sum;
3. reciprocal T area, live T thickness, divided hdiv, and rematerialized
   `e3t*hdiv`;
4. Kbb `r3t`, stored Kaa SSH, stored Kaa `r3t`, and `r1_Dt`;
5. Kaa-minus-Kbb r3, the left-associated per-level stretch, bracket, mask,
   incoming carry, and outgoing carry at every owned wet point;
6. final W, the W-only ZAD replay, and the independent
   source-order/live-total association.

The stored Kaa value must be scored against the exact production value used by
the candidate, not replaced by a recomputed continuity forecast. The later
post-solve `_g0` transport W is an explicitly separate comparison row.
Every row reports dtype, active count, unequal count, absolute maximum,
reference maximum, and normalized maximum.

## Frozen prediction and falsifiers

Prediction: Kbb U/V, metrics, live face and T thicknesses, Kbb r3, and `r1_Dt`
are bit-exact. The first non-bit input is the stage-1 stored Kaa SSH/r3 operand,
because NEMO reads that slot before the external solve while the current
production W path constructs a continuity forecast. Substituting only NEMO's
stored Kaa r3 into the otherwise live literal recurrence is predicted to make
W bit-exact and remove at least 90 percent of the W-only ZAD maximum on both
faces. The later post-solve transport W is predicted non-bit and must not be
substituted into stage-1 ZAD.

The prediction is **CONFIRMED** only if admission and NEMO-given-input replay
remain exact, every earlier registered input is bit-exact, stored Kaa SSH/r3
is the first non-bit input, its one-variable substitution makes final W
bit-exact, and the resulting ZAD replay removes at least 90 percent of both
Round-86 maxima. It is **REFUTED** if any earlier input is non-bit, Kaa is
bit-exact, substitution leaves any W cell unequal, either ZAD face removes
less than 90 percent, or the literal replay/calibration moves. Any refutation
is retained in the receipt and hands the measured first non-bit statement
forward.

Three differential controls must print `PLANT_FIRED` and exit nonzero: one U
flux-product ULP at the maximum residual, one stored-Kaa/stretch ULP at the
maximum residual, and one outgoing-carry ULP at the maximum residual. A plant
that touches a dry/undefined/irrelevant cell or does not change its named row
is a failed instrument.

## Conditional implementation and Rule 12

If and only if all inputs before the stored-Kaa selection are bit-exact and the
Kaa-only literal replay closes W exactly, test the smallest shared-source
change that supplies stage-1 ZAD from the pre-external, velocity-form WZV
boundary and leaves stage-2/3 stage-specific operands unchanged. The candidate
must not add a selector or stabilizer. If no production statement is locally
exact, land only the diagnostic and receipt.

| lane | frozen Round-87 disposition |
|---|---|
| GYRE kt2 WZV/ZAD | Score every row above; require given-input WZV/ZAD exactness and exact Kaa-only closure |
| GYRE kt=1..10 | If production changes, compare against `round85/bundle_kt1_10.json`; no AT-BAR row may leave the bar, first-over-bar may not move earlier, and every moved row must be registered |
| GYRE days 1..30 | Run only after the ladder passes; compare against `round85/bundle_day_gap.json` and register every day/field movement |
| LOCK_EXCHANGE-zco | If shared source changes, compare all 50 certified rows against the Round-85 post-history arm |
| OVERFLOW-zps | Prove the changed statement does not execute or compare all 50 certified rows against the Round-85 post-history arm |
| DINO | **SHARED-STATEMENT RISK:** DINO uses shared WZV/ZAD on leapfrog; numerically score it before any neutrality claim because its regional verdicts cancel by 96--98% per row |
| ORCA2 | **UNMEASURED-WITH-SPEC:** align pre/post-external Kaa SSH, Kbb velocity, every velocity/transport WZV input and carry, ZAD, cumulative RHS, and T/S/U/V/SSH on native masks through kt1..10 in fp64; reject any wet-input mismatch, AT-BAR loss, or earlier first-over-bar |

No configuration/default, coefficient, timestep, carried-state format,
restart contract, year harness, reconciliation gate, freshwater pair, #1484
guard, held manifest, NEMO source, or NEMO executable changes. Decisions
37--38 remain unchanged.
