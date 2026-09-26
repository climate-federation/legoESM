# Preregistration: literal Treguier column reductions, round 67

Date: 2026-08-30. Frozen before implementation and replay. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 66 stops at the first coefficient operand, `zn`, and shows that all
three independent column reductions differ at the same roundoff class:
`zn` maximum normalized error `2.7352e-8`, `zah` `7.6185e-8`, and `zhw`
`4.6767e-8`. NEMO `ldftra.F90:687-696` initializes scalar 2-D accumulators
and updates each once per `jk=1..jpk`; production uses three `jnp.sum` tree
reductions. The exact oracle-`zaeiw` U-face average is bit-exact, so there is
no independent face-builder defect.

Add `treguier_vertical_reduction_evaluation` with byte-preserving default
`"tree"` and DINO-faithful value `"nemo_left"` only on the two
`nemo_dino_kamm` cards. In `"nemo_left"`, initialize `(zn,zah,zhw)` to
`(0,0,5)` and use one `lax.fori_loop` over levels, updating the three values
with the exact source term order. The tree arm must retain its existing three
statements. Tests must establish default/generic byte identity, exact equality
to an explicit Python left fold, non-equality to a planted tree-reduction
case, JIT, and finite autodiff.

Replay the round-66 ladder and ordered row-8 scorer without changing bars.
`zn`, `zah`, `zhw`, `zRo`, `zaeiw`, `aeiu`, and row 8.8 must all pass before
8.8 is promoted. Then score rows 8.9 and 8.10 in order and continue to Redi.

Frozen round-66 receipt SHA-256:
`673d9dc978ac4c05a0c8a6995f7c9e1870d9502afedff9b765045df23014785f`.
