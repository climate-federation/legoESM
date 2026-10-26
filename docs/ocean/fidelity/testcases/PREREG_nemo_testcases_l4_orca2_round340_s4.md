# ORCA2 round 340 preregistration — exact standalone S4 acquisition

Frozen 2026-10-26 at base `d5d97cfc3121`. This round builds the operator-only
S4 acquisition and the legoESM composed card. It does not run NEMO.

The acquisition uses the admitted round-339 K0 recipe unchanged. Independent
instrumented twins record both ranks for steps 1--10: step entry, three RK3
stage exits, twelve source-ordered outer surface frames spanning S0-M
entry/exit, S1 entry/exit, S3 entry/exit, and the runoff, restoring,
freshwater-budget and rotation subframes of S2.  It also records S1's
turbulent and flux internals, every executing S0-F field read, and S3's
cover/freezing/clamp/exit internals. Admission predicts 240 outer-surface,
10 S1-turbulent, 10 S1-flux, 60 ice-if-internal, and 80 state records per
twin, plus a nonempty S0-F inventory, finite binary64 payloads, identical A/B
digests, and restart bytes identical to the observer-free K0 smoke. Any missing,
duplicated, reordered, truncated, checksum-bad, nonfinite, wrong-rank or
wrong-level frame refuses.

The scorer predicts 0/n unequal words at every registered surface boundary
before applying the existing unchanged 200-row rung-10 normalized predicate
(`max_abs / max(max_abs(reference), 1) <= 1e-15`). A one-ULP surface plant
and an over-bar ladder plant must refuse. The composed production call carries
all five Decision-175 selectors explicitly, reads its own physical S0-F inputs,
steps the existing three-dimensional rung-10 `LatLonCGridOceanModel` for steps
1--10, accepts no NEMO exchange record, and predicts finite N=1 gradients with
reverse directional derivative equal to forward within the existing binary64
gradient comparison precision.

The tool preflight must cover acquisition checklist 1--15: production-flag
nonzero FP census with arithmetic plant, declared extents, placement plants,
real ppsrc patching, K0 namelist and manifest pins, retained-run admission,
and syntax plus refusal/retention/success execution of every generated driver.
