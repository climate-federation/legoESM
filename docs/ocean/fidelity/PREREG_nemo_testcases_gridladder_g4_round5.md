# Preregistration — grid ladder G4 round 5 (gridladder_r21), NEMO-supplied forcing route

Frozen before any measurement of the route (Decision 123 = 1).

## What is built

The G4 card takes NEMO's own per-step surface fields through the route the
ORCA2 lane's ladders already use: the ORCA2 round-1 ladder gate's
`_surface_forcings` (NEMO's post-`sbc` fields -> the model's freshwater and
surface-forcing inputs), called unchanged, fed from the round-19 record's
`s` frame. No new forcing evaluator, no bulk change, no model change: the
ORCA2 and G4 cards consume forcing through identical model settings
(measured: their configurations differ in no forcing-consumption field;
river-runoff depth spreading is 0 on G4, which is NEMO's: ln_rnf_depth is
off in ORCA2's deck).

## Test (round-19 record, as-built deck; GIVEN-NEMO-ENTRY)

kt = 1, entry = NEMO's as-built initial state (the card's no-edits arm; the
record ran ln_tsd_dmp = F), forcing = NEMO's kt = 1 `s` frame. The card's
three RK3 stages are exposed and compared with NEMO's stage frames
(level-1 u, v, T, S; ssh, uu_b, vv_b), bit for bit.

## Predictions

- P1: stage 1 bit-identical on every row (stress enters the barotropic
  forcing, stp2d.f90:198-199; emp and rnf the sea-level forcing,
  stp2d.f90:242-245; emp the stage-1 tracer dilution, trasbc.f90:285).
- P2: stage 2 bit-identical on every row.
- P3: stage 3 T is the first row that may differ (qns, sfx and the runoff
  content enter only at stage 3, trasbc.f90:307-308, :323; solar at
  traqsr.f90:371-373; the river-mouth mixing, unmeasured, also acts there).
  PLAUSIBLE only.

## Falsifiers

- Any stage-1 or stage-2 row unequal: a field is read at the wrong program
  point or in the wrong form (named by which row moves first).
- Zeroing the forcing leaves stage 1 unchanged: the route is not consumed
  (non-vacuity control).
