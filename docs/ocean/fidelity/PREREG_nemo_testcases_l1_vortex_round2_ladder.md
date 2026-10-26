# PREREGISTRATION — VORTEX round 2, step D (the kt=1..10 ladder)

Frozen before a single ladder row is scored. The record exists and is admitted
(`phase3/vortex/round2`, restart byte-identical), and the initial-state
comparison has been run — those numbers are below and are therefore NOT
predictions, they are the measured entry condition this ladder starts from.
Everything under "Predictions" was written before the gate was pointed at this
card.

## What is already measured, and is not a prediction

The initial state, against BOTH records:

| field | cells unequal | of | worst |
|---|---|---|---|
| temperature | 0 | 37210 | exact |
| salinity | 0 | 37210 | exact |
| zonal velocity | 788 | 36600 | 2 last bits |
| meridional velocity | 788 | 36600 | 2 last bits |
| sea surface height | 104 | 3721 | 2 last bits |

Round 2's preregistration predicted the new record would reproduce round 1's
five rows exactly. It does, and more strongly than predicted: the two kt=1
records are **byte-identical**, so the equation-of-state switch provably did
not touch the initial state and the last-bit residual cannot be an artifact of
it. The residual is the compiled floor between two exponential functions.

## Predictions, frozen before scoring

The card has never executed a step, so a predicted error magnitude would be
theatre. What IS predicted, and is falsifiable:

1. **kt=1 is AT-BAR on temperature and salinity and DEBT on the three fields
   above.** The ladder's entry row is the initial state, which is already
   measured; a gate that disagrees with the table above is measuring something
   else and is wrong before it is informative.
2. **The first-over-bar step is kt=2 or later, never kt=1 for T or S.**
3. **If any field is over the bar at kt=2, the owner is the momentum path, not
   the tracer path.** This card has no surface forcing, no lateral diffusion,
   no bottom drag and a uniform salinity, so the only operators that can move a
   tracer in one step are advection and the vertical mixing coefficient. A
   tracer row that leaves the bar BEFORE the velocity rows would contradict
   that and is the finding.
4. **Salinity is UNINFORMATIVE from kt=2 on.** It is spatially uniform at 35
   everywhere, so its row cannot detect a transport or time-level error. The
   gate's own downgrade should say so; if it reports salinity AT-BAR as
   evidence, that is a vacuous control and is reported as one.

## Falsifiers

* The ladder's kt=1 rows disagree with the measured table above.
* A tracer field leaves the bar strictly before both velocity fields.
* The gate passes with its non-vacuity plant enabled (`--plant`), i.e. it
  cannot fail.
* Any row is reported AT-BAR on an empty or uniform mask without the gate
  saying so.

## Protocol, fixed before scoring

* The record's header is PARSED, never predicted (note BD). The existing gate
  hard-codes the two tanks' header tuples; adding a third hard-coded tuple
  would repeat the defect note BD was written for, so the tuple check is
  replaced by parsing for every case.
* NEMO's record carries one more vertical level than the card executes; the
  bottom record is the permanently dry dummy and is trimmed, exactly as the
  card's own initial-state gate already does.
* The bar is the gate's existing `1.0e-15` normalized maximum, unchanged. No
  new bar is invented for this card.
* Rows are per field per step, classified AT-BAR or DEBT by that bar, and the
  first-over-bar step is named whether or not it flatters the round.
* The two tanks and GYRE are re-run through the same gate afterwards; their
  numbers must not move.
