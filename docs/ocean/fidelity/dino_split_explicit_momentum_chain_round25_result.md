# Split-explicit momentum chain: round 25 result

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Verdict

`NEMO_SOURCE_ASSOCIATION_OWNS_FINAL_COEFFICIENT_DEBT`. The authoritative
offline artifact is
`/tmp/dino_split_explicit_momentum_chain_round25_association_factorial.json`,
SHA-256
`a9db28c458bb6e409ec23b74ea371b2d2e4aeaf1d84de322074ab72b46fca16e`.
It binds the round-24 artifact SHA-256
`3d0fa3148327daa36b7107a5b913b8f806090f5f9f76da93ff827c806c6872b4`
and reuses the retained round-22 streams; no NEMO process was launched.

The registered `2^3` factorial varied three arithmetic stages in NEMO source
order: the three-term `ff_f/e3f_vor` triads
(`dynspg_ts.F90:1331-1342,1358-1369`), surface-to-bottom left accumulation
(`:1344-1348,1371-1375`), and the post factors
(`:1349-1352,1376-1379`). The literal post-factor arm is the decisive main
effect, with aggregate coefficient-RMS effect `-1.5896271402e-15`. Vertical
left reduction contributes `-5.3769155482e-16` and interacts with the post
factor by `-4.6440618210e-16`; the triad main effect is
`-4.0962177932e-17`. These are binary64 association effects, not new physical
operands.

Both `T0V1P1` and the fully literal `T1V1P1` are bit-exact against all eight
stored NEMO `ffu/ffv` corner coefficients and against their U/V application:
normalized RMS `0`, maximum/NEMO-RMS `0`, and mismatch counts `0/9758` U and
`0/9868` V. The production control retains the ten-row DEBT topology from
round 24. Identity and the four-step `nextafter` plant pass.

## Retraction and control history

The provisional artifact SHA-256
`49df3aeeb5d27ff32ca2532f2649b09e0738795dc5aa4891f4773e57e83b4d62`
printed `SOURCE_ASSOCIATION_REFUTED` and is retracted. Its literal-post-factor
arm directly materialized the stored coefficient and then incorrectly sent it
through the checkerboard coefficient-inversion measurement operator a second
time. The authoritative scorer now keeps `P1` materialized; `P0` alone uses
the inversion. The scorer receipt names the retraction, so the withdrawn
verdict cannot be reproduced as an accepted result.

Earlier stopped attempts also exposed two control defects before a valid
disposition: a missing 2-D metric broadcast, and a demand that an independently
reconstructed association residual reproduce a nonzero production residual to
`1e-15` *relative to that residual*. The valid control admits the actual
round-24 production row by its bound SHA and requires the independent direct
construction to reproduce its ten-row DEBT topology. No ownership bar changed.

## Ordered disposition

The physical EEN coefficient composition and its final binary64 association
are now owned exactly. This authorizes a faithful, differentiable literal
coefficient builder on the two DINO fidelity cards, with generic cards
byte-pinned and red-capable conservation/JIT/gradient tests. That production
change is `DESIGNED_NOT_BUILT`: the current operator still reaches the generic
AL81 association. Consequently row 1.3 cannot yet be called fixed in
production.

The existing-dump inventory has the Coriolis-before-bottom fields and the
substep input velocities, but it has no executed after-bottom `zu_trd/zv_trd`,
live face drag/depth operands, or pre-boundary final-update output. Round 26
therefore preregisters one held NEMO bracket that captures bottom stress and
the immediately following vector update together. Row 1.4, rows 2--6, and the
free-surface-filter, momentum-RHS, and tracer-tail chains remain ordered behind
the literal coefficient implementation and that bracket.
