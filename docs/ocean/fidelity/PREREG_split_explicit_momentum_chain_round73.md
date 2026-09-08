# Preregistration: GM Rossby-radius composition, round 73

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 72 exonerates the final `zaeiw` association: literal source-order
evaluation with own `zRo/zah/zhw` is identical to production and remains red,
while substituting oracle `zRo` alone makes `zaeiw`, its U-face average, and
row 8.8 pass.  The direct `zRo` error is only `6.48431e-13` at its own bar but
is amplified by the square in `zaeiw`.

NEMO recomputes its Rossby denominator from the T-point latitude and applies
literal nested bounds (`ldftra.F90:776-781`):

`zfw=max(abs(2*omega*sin(rad*gphit)),1e-10)` and
`zRo=max(2e3,min(.4*zn/zfw,40e3))`.

legoESM currently consumes the stored `ff_t` and calls `clip`.  Reuse the
round-72 capture and score this frozen factorial at the unchanged `1e-12`
bar, propagating every arm through literal `zaeiw`, face averaging, and row
8.8:

- current `zRo` (required red downstream plant);
- literal nested bounds with stored `ff_t`;
- literal bounds with source-recomputed `2*omega*sin(lat_t)`;
- literal bounds with oracle `zn` and stored `ff_t`;
- literal bounds with both oracle `zn` and source-recomputed Coriolis;
- direct oracle `zRo` (must pass the downstream chain).

If source-recomputed Coriolis alone passes, own `ZFW_COMPOSITION`; if oracle
`zn` alone passes, own `ZN_PRECURSOR`; if only their joint arm passes, retain
`ZN_X_ZFW_COMPOSITION`.  If the all-substitution arm fails, stop invalid.
No production change is admitted before classification.

Frozen round-72 receipt SHA-256:
`1abf7718e8dc1fc4f75d23295ebaaf46c368ffa07d8e56467577f1abb7201230`.
