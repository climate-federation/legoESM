# Preregistration: GYRE round 66 Krhs/LDF discriminator

Date: 2026-09-12. Frozen after the five-operand walk named `Krhs`, and before
capturing or changing legoESM's GM/Redi result. The prior tracer-operand
prediction is REFUTED and is not retroactively changed here.

## Observed boundary and source reading

The admitted round-64 record's post-hoc oracle increments are: LDF T maximum
`1.165238069992448e-8 s-1` and S maximum `4.836453217186004e-10 s-1`.
These magnitudes match the measured full-live-Krhs discrepancies
(`1.165835192724526e-8`, `4.837171800157751e-10`) and motivate, but do not
confirm, this lane.

The compiled active program calls `tra_qsr` then `tra_ldf` at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:932-951`.
Resolved configuration selects standard iso-neutral Laplacian at
`round64/oracle_krhs_split/ocean.output:649-656,1105`.
`traldf_iso.f90:184-187` differences `pt(Kbb)`; `:241-247` constructs
Kmm-metric face fluxes; and `:287-305` adds their divergence to `pt(Krhs)`.

legoESM computes the corresponding shared GM/Redi tendency from Kbb tracers at
`ocean_model_latlon_cgrid.py:7082-7101,7127-7365`. It currently adds that
tendency to `T_mid/S_mid` at `:7474-7477`, while the later WS stage-3 call
restarts from `state.T/S` and consumes `_stage_source_rates` at `:7639-7677`.

## Prediction, discriminator, and control

Prediction: the GM/Redi tendency is absent from the WS stage-3 Krhs
accumulator. Capture the existing production GM/Redi return without changing
it. For T and S, score (1) live Krhs against each admitted cumulative boundary,
(2) captured GM/Redi against `after_ldf-after_qsr`, and (3) live Krhs plus that
captured tendency against `after_ldf`.

CONFIRM if the closest live boundary is `after_qsr`, adding only the captured
GM/Redi tendency reduces T content-equivalent maximum below `2e-6` and by at
least 5x, and the reciprocal removal from oracle reproduces the removed
O(1e-3) content difference. REFUTE otherwise; land no LDF routing change.
Exact unequal counts, RMS, maxima, and signed means are reported on 18,000 wet
cells. A one-ULP captured-tendency plant must change the exact census.

If confirmed, the one shared WS implementation folds the existing GM/Redi rate
into stage-3 `_stage_source_rates`; it does not create a second LDF operator.
GYRE kt=1..10 and days 1..30, LOCK_EXCHANGE, OVERFLOW, DINO separation, and
ORCA2 specification remain governed by the parent round-66 preregistration.
