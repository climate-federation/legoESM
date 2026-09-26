# Preregistration: exact GM sqrt production certification, round 75

Date: 2026-08-30. Frozen after implementation and before production replay.
Session `01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 74 owns the remaining row-8.8 error to legoESM's `1e-30` forward floor
inside `sqrt(MAX(rn2b,0))`.  Its exact-forward arm makes `zn` bit-exact and
passes `zRo`, `zaeiw`, `aeiu`, and row 8.8.  Production now exposes the
selector `treguier_sqrt_evaluation`: global/default `guarded_floor` retains
the old bytes, while only `nemo_dino_kamm` and `nemo_dino_kamm_mlf` select
`nemo_forward_exact`.  A custom JVP supplies derivative zero at non-positive
N² while preserving NEMO's exact forward zero.

Replay the full production row-8 chain and coefficient ladder under unchanged
bars.  Reconstruct the legacy guarded-floor `zn` independently from the exact
captured operands; it must remain red after propagation and is the planted
violation.  Production must pass `zn`, `zRo`, `zaeiw`, `aeiu`, and row 8.8,
then score rows 8.9 and 8.10 in order.  Promote the tracer-entry chain only if
all rows 8.3--8.10 pass; otherwise stop at the first ordered failure.

Frozen round-74 receipt SHA-256:
`56db4716cba582654fbd7bb55178a699b55678a1afdea9d8d8fe3cc670eea6fb`.
