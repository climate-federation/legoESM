# Preregistration: tracer-entry row 8 replay, round 55

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 54 closes the Kmm execute/undo and Asselin arithmetic given the retained
NEMO transport, while preserving production `Hu_avg/Hv_avg` as upstream debt.
The earlier lateral-tracer lane stopped at production subrow 8.3 before the
momentum fixes. Reuse its admitted held row-8 streams and execute one current
single-pass DINO MLF step. Capture the active `add_bolus_to_advecting_flux`
handoff and score, in source order, `uu(Kmm)`, `e2u`, live `e3u(Kmm)`, their
product, Eulerian pU, GM pU, total pU, and the temperature upstream flux.

Subrows 8.3--8.6 use the unchanged pointwise whole-column bar `1e-15`;
8.7--8.10 use the unchanged accumulated-flux bar `1e-12`. Stop at the first
red subrow. Identity, wet-point, meridional-roll, and sign-flipped-GM plants
must fire; the production hook must be restored; all registered populations
must be nonempty and finite; round 54 and every held stream/source are
hash-bound. If all subrows pass, release Redi T/S and the complete tracer
state tail. If 8.3 fails, retain production `Hu_avg/Hv_avg` as the ordered
owner rather than reopening GM or FCT.
