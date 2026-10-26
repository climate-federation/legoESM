# Split-explicit momentum chain: round 24 result

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Verdict

`JOINT_COMPOSITION_PARTIAL`. The binding artifact is
`/tmp/dino_split_explicit_momentum_chain_round24_qco_factorial.json`, SHA-256
`3d0fa3148327daa36b7107a5b913b8f806090f5f9f76da93ff827c806c6872b4`.
It reuses the exact round-22 run manifest; no NEMO process was launched.

The registered 2x2 identifies a strongly interacting composition. Aggregate
eight-coefficient normalized RMS is `2.7995854532e-4` in `F0Q0`,
`9.4097231794e-5` with NEMO F-point f alone, `2.4903028249e-4` with literal
QCO thicknesses alone, and `2.2092430508e-15` with both. Signed factorial
effects are `F=-1.8586131353e-4`, `Q=-3.0928262828e-5`, and
`F×Q=-6.3168968963e-5`. Therefore neither substitution is an independent
owner: NEMO's F-point latitude and live QCO face/F thickness construction own
the physical coefficient composition jointly.

All upstream controls pass. Reconstructed `r3u/r3v` match their deterministic
dumps at normalized RMS `7.99e-17/7.73e-17` and maximum/RMS
`4.14e-16/4.17e-16`. The four consuming-face metric operands `e1u/e1v/e2u/e2v`
are bit-exact on 9,758 U or 9,868 V points. `F0Q0` reproduces round 22 and
`F1Q0` reproduces round 23. Identity passes and the four-step `nextafter`
plant fails.

The joint arm misses only the maximum pointwise gate. Its eight coefficient
normalized RMS errors are `2.56e-16`--`3.25e-16`, but maximum/RMS is
`1.12e-15`--`1.68e-15`. Applied U/V output RMS is
`2.29e-16/1.99e-16`, while maximum/RMS is `3.18e-15/4.93e-15`.
This is a binary64 association residual, not an unresolved physical operand,
but the campaign bar is unchanged and the row is not called exact.

## Ordered disposition

NEMO's source path is now bounded to `dynspg_ts.F90:1331-1342` (three-term
`ff_f/e3f_vor` association), `:1344-1347` and `:1371-1374` (vertical left
accumulation), and `:1349-1352` plus `:1376-1379` (post-factor association).
The registered round-25 arithmetic factorial is the next legal measurement.
Implementing a literal, differentiable coefficient builder plus its
association selector is larger than this round, so no production default is
changed. Row 1.3 bottom stress, row 1.4, rows 2--6, and the free-surface,
momentum-RHS, and tracer-tail chains remain ordered-blocked behind that exact
arithmetic disposition.
