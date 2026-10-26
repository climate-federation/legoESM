# ORCA2 round 340 preregistration — exact standalone S4 acquisition

Frozen 2026-10-26 at base `d5d97cfc3121`. This round builds the operator-only
S4 acquisition and the legoESM composed card. It does not run NEMO.

The acquisition uses the admitted round-339 K0 recipe unchanged. Independent
instrumented twins record both ranks for steps 1--10: step entry, three RK3
stage exits, eight source-ordered outer surface frames spanning S0-M
entry/exit, S1 entry/exit, S3 entry/exit, and S2 entry/exit, all executing
S0-F fields, and S3's cover/freezing/clamp/exit internals. Admission predicts
160 outer-surface, 60 ice-if-internal, and 80 state records per twin, plus a
nonempty S0-F inventory, finite binary64 payloads, identical A/B digests, and
restart bytes identical to the observer-free K0 smoke. Any missing,
duplicated, reordered, truncated, checksum-bad, nonfinite, wrong-rank or
wrong-level frame refuses.

The scorer predicts 0/n unequal words at every registered surface boundary
before applying the existing unchanged rung-10 per-row exact predicate. A
one-ULP plant in every group must refuse. The composed production call carries
all five Decision-175 selectors explicitly, reads CORE and Decision-171 files
directly, accepts no NEMO exchange record, and predicts finite N=1 gradients
with reverse directional derivative equal to forward within the existing
binary64 gradient comparison precision.

The tool preflight must cover acquisition checklist 1--15: production-flag
nonzero FP census with arithmetic plant, declared extents, placement plants,
real ppsrc patching, K0 namelist and manifest pins, retained-run admission,
and syntax plus refusal/retention/success execution of every generated driver.
