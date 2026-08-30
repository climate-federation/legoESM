# Preregistration: row-1.3 EEN source-association factorial, round 25

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Frozen before constructing or scoring
any literal-association arm. Bound predecessor: round-24 artifact SHA-256
`3d0fa3148327daa36b7107a5b913b8f806090f5f9f76da93ff827c806c6872b4`.

## Question and frozen operands

Round 24 leaves no physical operand above the bar: its joint NEMO `ff_f` plus
live-QCO-thickness arm has coefficient RMS `2.56e-16`--`3.25e-16`, but a few
points miss the maximum/RMS bar by up to `1.68e-15`; application raises that
maximum to `4.93e-15`. The next measurement changes arithmetic association
only. It reuses the retained round-22 coefficient/velocity dumps and round-24
operand construction. No NEMO run or new dump is authorized.

NEMO's executed ordering is:

1. triads `(ff_f/e3f + ff_f/e3f) + ff_f/e3f` at
   `dynspg_ts.F90:1331-1342,1358-1369`;
2. `coefficient = coefficient + e3face*e3neighbor*mask*triad` from surface to
   bottom at `:1344-1348,1371-1375`;
3. source-ordered post factors `r1_12*r1_e*r1_h*neighbor_metric*coefficient`
   at `:1349-1352,1376-1379`.

The scorer runs the full `2^3` offline factorial over `T` (literal three-term
triad association), `V` (literal surface-to-bottom left accumulation), and `P`
(literal post-factor association). A factor-off arm retains the production
JAX association for that stage while consuming identical binary64 operands.
It reports all main effects, all pair interactions, and the three-way
interaction on aggregate eight-coefficient RMS. The production `T0V0P0` arm
must reproduce round 24 within `1e-15` relative. Identity and a decisive
repeated-`nextafter` plant must pass.

Every coefficient and applied output retains the POINTWISE bar: normalized RMS
and maximum/RMS both `<=1e-15`; bit mismatch counts are diagnostic. An arm may
own the arithmetic residual only if all eight coefficients and both outputs
are at bar. If `T1V1P1` is exact, disposition is
`NEMO_SOURCE_ASSOCIATION_OWNS_FINAL_COEFFICIENT_DEBT`; if it removes at least
90% but misses any maximum bar, disposition is `SOURCE_ASSOCIATION_PARTIAL`;
otherwise it is `SOURCE_ASSOCIATION_REFUTED`. Only the exact verdict authorizes
the faithful literal coefficient builder. Generic cards remain byte-pinned,
and the existing conservation plants remain mandatory.

Rows later than the live EEN coefficient remain ordered-blocked until this
factorial is adjudicated. Any future held instrumentation uses the canonical
`__MEASURED_<name>__` SLOT prefix, but round 25 itself requires none.
