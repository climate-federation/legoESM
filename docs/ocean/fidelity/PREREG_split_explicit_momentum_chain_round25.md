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

## Control-failure amendment before a valid factorial

Two committed attempts to reconstruct `T0V0P0` stopped at the frozen control,
before emitting an artifact or disposition.  The direct source stencil put all
eight coefficient RMS rows closer to NEMO (`2.10e-16`--`2.21e-16`), but differed
by 16--35% from round 24's nonzero `2.56e-16`--`3.25e-16` checkerboard-inversion
residues. Forwarding it through the same four-pattern solve reduced that gap to
3--15%, but cannot make a separately evaluated stencil reproduce the exact
roundoff of the production JAX operator plus inversion to `1e-15` *relative to
the residual*.

The valid factorial therefore admits `T0V0P0` verbatim from the SHA-bound round
24 artifact. This is the actual production control, not a reconstruction. The
separately materialized `T0V0P0` remains in the receipt and must put all eight
coefficients and both literal-applied outputs at the unchanged POINTWISE bar;
otherwise the factorial stops. Other arms use that one committed materializer.
No physics bar, arm, or ownership rule changes. This amendment is frozen before
the first factorial invocation permitted to emit a disposition.

### Direct-baseline control correction

The first hash-admitted invocation stopped because the amendment above
mistakenly required the independently materialized *off/off/off* arm to be at
bar. That contradicts the experiment: off/off/off is precisely the association
DEBT control. Its eight coefficient maxima (`1.11e-15`--`1.40e-15`) and two
output maxima (`3.98e-15/4.93e-15`) correctly remain DEBT. Before any valid
disposition, the independent control is corrected to require the same ten-row
DEBT topology as the SHA-admitted round-24 baseline. The admitted values remain
the quantitative control. The full-literal arm still must pass the unchanged
POINTWISE bars; no ownership condition changes.
