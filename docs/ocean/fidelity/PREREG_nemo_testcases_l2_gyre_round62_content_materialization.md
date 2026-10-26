# GYRE round 62 content-statement materialization preregistration

Date frozen: 2026-09-12. Parent candidate: `c03458508a76`.

The plain algebraic regrouping prediction is **REFUTED**: XLA emitted the same
content bits and kt=3 result. Before the next run, the only new arm is frozen:
materialize the advection division, Krhs accumulation, each thickness product,
and final sum with the existing `nemo_source_round` helper. CONFIRM only if the
live content T/S and independently advanced kt=3 T/S are bit-exact; REFUTE if
any cell differs. Coefficient materialization remains a separate statement.
