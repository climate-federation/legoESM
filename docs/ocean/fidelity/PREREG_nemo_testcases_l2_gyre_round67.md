# Preregistration: NEMO-testcases L2 GYRE round 67 LDF order

Date: 2026-09-12. Frozen before the round-67 cumulative substitution,
pre-solve override, production edit, and trajectory measurements.

## Source-order claim

The admitted record is round 64's kt=2 stage-3 Krhs record, with the admission
census and controls fixed by the round-64 receipt. NEMO clears Krhs, calls
`tra_adv`, then `tra_sbc_RK3` at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-868`.
At stage 3 it calls `tra_qsr`, `tra_ldf`, and then `tra_zdf` at
`:917-965`. The implicit solve constructs its coefficient from `avt` plus
`ah_wslp2` at `trazdf.f90:416-443`, constructs the matrix at `:463-479`, and
constructs the content right-hand side as
`e3t(Kbb)*T(Kbb) + p2dt*e3t(Kmm)*Krhs` at `:547-565`.

legoESM constructs the stage-3 surface/solar source in concentration-rate form
at `ocean_model_latlon_cgrid.py:6257-6348`. It computes the GM/Redi rate from
Kbb operands at `:7048-7127,7340-7460`, then adds `dt*dT_gm`/`dt*dS_gm` to
the intermediate concentration at `:7474-7480`. The WS-RK3 call later restarts
from the step-entry tracer and consumes only `_stage_source_rates` at
`:7639-7677`; it constructs content at `:1905-1917`, and the implicit solve
consumes that separate content at `:8038-8150,10417-10438`. Therefore the
GM/Redi increment is not applied after the solve: it is applied in
concentration form before the WS call and is then replaced. Moving that
existing increment into the stage-3 pre-solve content is NEMO's operator order.

No new card field selects this order. The existing `rk3_ws` dispatch selects
the distinct stage program; within that program, a configured GM/Redi operator
must enter stage-3 Krhs unconditionally. DINO uses the separate modified-
leapfrog program and is not selected by this statement.

## Cumulative substitution and falsifiers

Let `A`, `B`, `Q`, and `L` be stage-3 advection, surface-boundary, solar, and
lateral-diffusion concentration increments. Capture the already-executing live
QSR and LDF results without changing their returns. Rebuild the routed live
Krhs as `A_live+B_live+Q_live+L_live`, then replace NEMO cumulative fields in
compiled order, one boundary at a time:

1. post-SBC: `after_sbc_NEMO + Q_live + L_live`;
2. post-QSR: `after_qsr_NEMO + L_live`;
3. post-LDF: `after_ldf_NEMO`.

Score T and S rate and content unequal count, RMS, maximum absolute difference,
and signed mean on the 18,000 wet cells. The baseline and each arm use the same
five-operand content statement. The oracle post-LDF arm must reconstruct the
recorded content exactly; a one-ULP wet-cell plant must exit nonzero.

Prediction: after routing LDF, the remaining T/S content is owned by the
already-measured complete-FCT Krhs debt, not by a missing surface/QSR content
piece, a thickness association, or K33 placement. CONFIRM only if routed live
Krhs is nearest post-QSR, the post-SBC substitution does not remove the floor,
and the post-QSR substitution leaves only the measured live-versus-oracle LDF
increment error. REFUTE if a surface or QSR substitution removes a discrepancy
larger than the FCT floor, if post-LDF is not exact, or if the live source
cannot be decomposed into the active SBC+QSR calls. K33 cannot own a pre-solve
content discrepancy; it remains a candidate for the solved-state remainder
and is REFUTED only if the recorded matrix/solution replay shows its placement
already agrees.

## Re-derived landing criterion

The admitted round-66 floor is not `2e-6` content. The measured routed Krhs
debt is T `6.198289884214600e-11 K s-1` and S
`5.135349555322604e-12 s-1`; NEMO's `p2dt=14400 s`, so the concentration
floor before vertical redistribution is approximately `8.93e-7 K` and
`7.40e-8`. With the recorded Kmm thickness, its exact content manifestation is
T `5.954039670541533e-5 K m` and S `7.651457963220310e-6`.

The routing statement is eligible only if:

- its captured pre-solve content equals the independently constructed
  `live Krhs + captured LDF` arm cell-for-cell, apart from a separately reported
  source-association rounding row;
- its maximum content discrepancy is no larger than the frozen manifestations
  above plus that measured association rounding, and improves T by at least
  20x from `1.679392691670500e-3`;
- a pre-edit override of that same content through the production implicit
  solve predicts the post-solve kt=2 and next-step kt=3 T/S floors, and the
  implementation reproduces those predictions without an earlier over-bar
  boundary.

No scalar tolerance is invented for kt=3: the frozen target is the exact
pre-edit content-override result, scored with exact census and maximum
difference. The existing kt=3 baseline is T/S
`1.627511418e-4/6.327755180e-6`.

## Rule 12 and scope

After the edit, measure GYRE kt=1..10 and days 1--30 against the recorded
decision-36 after arm. Register every moved row and require the first-over-bar
boundary not to move earlier. LOCK_EXCHANGE and OVERFLOW execute the same
WS-RK3 program but resolve `ln_traldf_OFF=.true.` and zero tracer diffusivity;
their corresponding legoESM cards have `gm_redi=None`, so the statement is
predicted bit-inert. Confirm with their kt rows and exact before/after artifact
comparisons. DINO's leapfrog statement must be bit-inert. ORCA2 remains
UNMEASURED only with the exact command, card, oracle records, fields, and
pass/fail rule recorded in the receipt.

No configuration/default, carried state, NEMO source/build/run, year harness,
reconciliation gate, freshwater pair, #1484 guard, or held round-60/62 patch is
changed. A failed prediction is recorded as REFUTED and no physics edit lands.

## Pre-measurement control amendment after refused run

The first committed-instrument invocation exited nonzero before writing a
report. The prediction that the post-LDF *single-Krhs substitution* would make
content exact is **REFUTED**: round 66 had already measured that arm's separate
live-Kmm-thickness association residue, T `1.359695e-10` and S
`6.366463e-12` content units. The instrument had conflated two controls.

Before any successful measurement, the calibration is split without changing
an acceptance threshold: (1) the all-oracle five-operand statement must be
bit-exact, as the preregistration intended to test record integrity; and (2)
the post-LDF single-Krhs substitution must reproduce round 66's already-frozen
Krhs-substitution row exactly. The refused run produced no citable physics
number. The cumulative owner prediction and landing floors above are unchanged;
the retracted exactness claim remains printed here as required by Rule 11.

## Pre-trajectory correction

The Rule-12 paragraph above originally said “decision-36 before arm.” That was
a transcription error, corrected here before any trajectory measurement: the
user-fixed comparator is decision 36's **after** arm, produced at
`f78547b752f733c4d86f024df7effc6f5b2e376a`. A source diff from that revision
through round 66 is empty for both the shared ocean stage program and the GYRE
card, so it is also the same production baseline as this round's parent. The
original wording and this correction are both retained in Git history.

## Pre-successful implementation-gate calibration

The first post-edit invocation refused before writing a report because its
check demanded that content reconstructed by dividing the captured content to
Krhs and multiplying it back be cell-exact. That contradicts the already-
preregistered “apart from a separately reported source-association rounding
row” qualification. Before a successful invocation, the check is split:
report captured-content versus reconstructed-content as that association row;
independently score captured production content against the oracle; require its
maximum no larger than the frozen routed maximum plus the measured association;
and retain exact equality of the complete kt3 metric row to the frozen pre-edit
override. The refused run wrote no report and supplied no citable number.

## Association follow-up after a persisted REFUTED implementation

The first persist-capable implementation run exited nonzero with status
**REFUTED**. Routing LDF through `stage_source_rates[2]` alone leaves the WS
helper's old association
`(h(Kbb)*T - dt*flux_div) + dt*h(Kmm)*source`. The independently constructed
pre-edit prediction used NEMO's compiled association
`h(Kbb)*T + dt*h(Kmm)*(-flux_div/h(Kmm) + source + LDF)` from
`trazdf.f90:547-565`. The two differ by reassociation: measured production
versus reconstructed routed content is T `2.273736754432e-13` and S
`1.818989403546e-12`; the kt3 implementation-versus-override maxima are T
`7.105427357601e-15` and S `2.131628207280e-14`. The frozen exact kt3 criterion
therefore correctly rejected the first edit.

Before changing the shared statement, preregister the only source-literal
follow-up: construct stage-3 Krhs as
`-flux_div/h(Kmm) + stage_source_rates[2]` and form content once as
`h(Kbb)*T + dt*h(Kmm)*Krhs`, for every `rk3_ws` caller. This is not a card arm.
CONFIRM only if GYRE production content and the complete kt3 T/S metric rows
are exactly the frozen pre-edit override, and all other landing criteria hold.
REFUTE on any difference.

This universal association also executes when LDF is off, so the earlier
prediction that LOCK_EXCHANGE/OVERFLOW would be bit-inert applies only to the
LDF addition and no longer discharges the combined patch. Their prior
bit-identity prediction is retained but **withdrawn before measurement** for
the association statement: enumerate every moved row, require the ULP move
gate and unchanged/not-earlier first-over-bar. DINO remains inert because it
does not call the WS helper. ORCA2 remains UNMEASURED-with-spec.
