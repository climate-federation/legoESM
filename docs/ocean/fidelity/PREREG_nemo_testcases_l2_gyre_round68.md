# Preregistration: NEMO-testcases L2 GYRE round 68 production-FCT content

Date: 2026-09-12. Frozen before the production-advection-content measurement,
any production edit, and every trajectory measurement.

## Immutable record and compiled statement

The admitted oracle remains
`round64/oracle_krhs_split/oracle_krhs_split_kt00000002.bin`, produced by
`3b3b045bd9e03b60330204e7590e4c4470b7a0ca`. Admission must reproduce 43/63
exact inherited records, 20 changed records, and all 132 admitted consumed
values. Its wrong-commit, truncation, record-stamp, and one-ULP plants must exit
nonzero.

The compiled GYRE stage clears `Krhs`, accumulates advection and SBC at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-868`, then at
stage 3 calls QSR, LDF, and ZDF in that order at `:917-965`. The active
isoneutral operator reads `T/S(Kbb)` at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:182-204`, builds
Kmm-metric fluxes at `:227-246`, and adds their divergence to `Krhs` at
`:257-305`. The implicit update forms
`e3t(Kbb)*T(Kbb) + p2dt*e3t(Kmm)*Krhs` before its forward sweep at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:547-565`.

legoESM's production WS helper already materializes its exact stage-3 FCT
advection content as `h(Kbb)*T(Kbb)-dt*flux_div` and then adds the source
content at `ocean_model_latlon_cgrid.py:1991-1997`. It returns both arrays at
`:1918-1922`, and the production step retains them at `:7635-7683`. The
existing WRITE-only `expose_stage3_advection_content` seam at `:1273-1283` and
`:8031-8040` proves that this is the production array; no recomputed FCT
divergence is admissible for this round's prediction.

## Frozen prediction and falsifiers

Capture, in the ordinary compiled production step, the returned stage-3
advection content `Cadv`, the existing stage-3 source rate `Rsrc`, the already
executed GM/Redi rate `Rldf`, and the admitted live `p2dt` and `e3t(Kmm)`.
First require the unmodified production content to be cell-exact to
`Cadv + p2dt*e3t(Kmm)*Rsrc`. Then construct exactly once, in that association,

`Cpred = Cadv + p2dt*e3t(Kmm)*(Rsrc + Rldf)`.

Prediction: `Cpred` is the exact output of routing the existing LDF tendency
into the WS stage-3 source tuple, and its post-solve kt=3 T/S metric rows are
the exact candidate trajectory. CONFIRM the pre-edit prediction only if the
baseline reconstruction has zero unequal wet cells, the one-ULP content plant
changes exactly one wet cell, and `Cpred` improves the admitted T content by at
least 20x while remaining no larger than the round-67 routed floor
`5.954039670541533e-5 K m` plus only a reported last-bit association term.
REFUTE otherwise and land no production change.

After a source-literal routing edit, CONFIRM implementation only if captured
production content equals `Cpred` with zero unequal wet cells, its complete
kt=3 T and S metric dictionaries equal the frozen pre-edit override
dictionaries exactly, and applying the same override changes zero wet cells.
Any unequal cell, including an equal maximum with a different census or RMS,
REFUTES the edit. The round-67 separately recomputed-FCT prediction remains
REFUTED and is not reused.

## Eligible implementation and Rule 12

Only if confirmed, move the already-computed signed GM/Redi T/S rate from the
discarded WS concentration update into `_stage_source_rates[2]`. Keep every
non-WS program's current update. Do not add a second LDF operator, alter K33,
change the FCT statement, introduce a card/default/threshold, or change carried
state. The sign convention is positive rate into tracer concentration and
positive `Krhs` into NEMO's content RHS, as compiled at
`traldf_iso.f90:287-305` and `trazdf.f90:547-565`.

Run GYRE kt=1..10 through the required phase-3 ladder gate and compare every
moved row against decision 36's recorded after arm
`f78547b752f733c4d86f024df7effc6f5b2e376a`. No AT-BAR row may leave the bar
and the first-over-bar boundary may not move earlier. Run days 1..30 from the
same production baseline with the fixed daily harness and score every day.

LOCK_EXCHANGE and OVERFLOW execute the shared WS helper but resolve LDF off and
`gm_redi=None`; require exact before/after artifact identity for kt=1..10 and
register any move. DINO uses the separate modified-leapfrog program; require a
source-path proof plus its existing execution gate or byte-identity artifact.
ORCA2 may remain UNMEASURED only with this executable specification: resolve
its native card, record stage-3 post-SBC/QSR/LDF Krhs and pre/post-ZDF T/S for
kt=1..10, run the cumulative/content gate and independent trajectory gate, and
require exact statement replay, every moved row registered, no AT-BAR loss,
and no earlier first-over-bar boundary.

No NEMO source/build/run, configuration choice, carried-state change, year
harness, reconciliation gate, freshwater pair, #1484 guard, or held manifest
is modified. A failed prediction is retained as REFUTED.

## Pre-successful-run control correction

The first invocation exited nonzero before writing a report. It attempted to
read a production-versus-recomputed association row from the archived
round-67 pre-edit JSON, but that historical report predates the row. No physics
number was emitted or retained. Before the first successful measurement, the
control is corrected to calculate the already-preregistered last-bit
association directly between this run's production-content prediction and its
round-67 recomputed routed content. The frozen prediction, floor, and
falsifiers above are unchanged.
