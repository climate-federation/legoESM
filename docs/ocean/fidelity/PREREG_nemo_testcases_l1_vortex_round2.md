# PREREGISTRATION — VORTEX card, round 2

Frozen before the round-2 record is acquired and before any kt=1..10 row is
scored. Lane tip `5301122fe2ef`, with round 1's eleven commits forward-ported
onto it (round 1 was never landed; it sits on an older tip).

## What this round changes, and what it must not

Two decisions (operator note BG):

* **D69** — VORTEX runs NEMO's simplified equation of state, the one narrow
  exception to the campaign's TEOS-10. The card selects it with the shipped
  `&nameos` coefficients, and the oracle deck stops switching it.
* **D70** — the Courant-dependent implicit vertical advection stays OFF, as
  the shipped deck leaves it. No change; round 1 already matched it.

And one thing round 1 left open: the case's Coriolis operator, which round 1
could not express and declared as a gap. Round 2 transcribes it.

**Additivity is the binding constraint.** No other card may move. That is
measured, not asserted: a field-by-field diff of GYRE's, LOCK_EXCHANGE's and
OVERFLOW's resolved configurations across this round must report exactly one
field added, none removed and none changed, and the added field must be unset
on all three.

## Predictions, frozen

| row | prediction |
|---|---|
| initial-state temperature, cells unequal | 0 |
| initial-state salinity, cells unequal | 0 |
| initial-state zonal velocity, cells unequal | 788 of 36600, none worse than 2 last bits |
| initial-state meridional velocity, cells unequal | 788 of 36600, none worse than 2 last bits |
| initial-state sea surface height, cells unequal | 104 of 3721, none worse than 2 last bits |
| the same five rows against the NEW, S-EOS record | IDENTICAL to the above |
| certified card digests | change exactly once, by the one added configuration field, and never again |

**Why the velocity and height rows are not zero, stated before the record is
read.** Round 1 predicted a bit-exact initial state on all five fields. That
prediction is refuted, and the refutation was measured against round 1's own
record before this preregistration was written. The residual is the compiled
floor between two exponential functions: every unequal cell differs by one or
two last bits, and they sit in the far tail of the eddy's Gaussian, where the
unequal heights run down to 4.5e-92 while the metre-scale centre is exact. The
card already declares that it evaluates transcendentals with the system
library.

**Why the new record must reproduce those counts exactly.** Nothing in the
initial state reads the equation of state. Temperature is built by inverting
it, but from the namelist coefficient, which is read whichever law is
selected; velocity and height are analytic. So switching the deck's equation of
state must leave all five rows untouched. If it does not, the transcription of
the initial state is wrong somewhere this round did not look, and that is the
finding, not the ladder.

## Falsifiers

* Any certified card's resolved configuration differs in a field this round did
  not add.
* The new record's initial-state rows differ from the table above.
* A cell unequal by more than two last bits on any field.
* The card's Coriolis operator can be selected in any composition that was not
  read off NEMO's source, or on a mesh where the dropped metric term is live.

## The kt=1..10 ladder

Scored only after the record is admitted, the tanks' way: per-field rows, each
classified at-bar or debt, and the first-over-bar step named. Not preregistered
with numbers, because this card has never executed a step and a predicted
number with no basis is theatre. What IS preregistered: the ladder is scored on
the S-EOS record, not round 1's, and the first step at which any field leaves
the bar is reported whether or not it flatters the round.
