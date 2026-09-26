# Preregistration: hardened continuity-to-recurrence admission, round 15

Date: 2026-08-29. Frozen before measurement.

## Admission correction

Adversarial review found that the first round-14 receipt did not mechanically
bind its required round-13 release artifact, and that round 13 did not enforce
checkout-local production imports or hash the mesh and restart named by its
preregistration.  The first round-14 receipt is therefore downgraded to
`UNBOUND_DIAGNOSTIC`; none of its row verdicts is citable until this round
reproduces and binds them.

Before any rerun, the round-13 scorer is hardened to require every production
import beneath the clean measured checkout, hash every imported production
module, hash the exact mesh and restart, enforce `d180`/E3T=`both`, and retain
all prior session, dump, population, control, CPU/fp64, and association-artifact
bindings.  No calculation, candidate, metric, population, or bar changes.

## Frozen sequence

1. At one clean committed package, rerun the hardened round-13 scorer on the
   existing round-9 SLOT output.  Rows 9.1--9.7 must all be `AT BAR` and its
   disposition must release 1.4.
2. At that same commit and session, rerun the unchanged committed round-6
   full-recurrence wrapper on the `d180` lane with held slow forcing, SSH
   forcing, and continuity localization enabled.
3. Pass both artifacts to the committed round-15 adjudicator.  It SHA-binds
   them, verifies identical package commit/session, verifies the hardened
   production/input bindings and controls, requires the round-13 release, and
   only then reports the recurrence's first ordered stop.

The recurrence bars and order are unchanged from round 14: authoritative
`forcing_only` row 1.2 POINTWISE `1e-15`, then row 1.3 and row 1.4
ACCUMULATING `1e-12`, every field required at bar.  The exact-seed arm remains
targeting-only.  Rows 2--6 and the climate GPU arms are released only if rows
1.2--1.4 all clear.

Identity and planted-violation controls, exact held forcing, restored
monkeypatches, finite populations, clean-before/after, CPU/fp64, and no escaped
imports remain mandatory.  This round runs no NEMO process, GPU, `mpirun`, or
push and changes no production physics.
