# Certified GYRE reverse-mode gradient gate — round 3 preregistration

Date frozen: 2026-10-11

This file freezes round 3 before any new gradient or VJP-bisect measurement.
The numerical protocol and bars remain those registered in
`nemo_testcases_l2_gyre_gradgate_round1_prereg.md`: the certified `GYRE-zco`
card, CPU fp64/libm, production `LatLonCGridOceanModel._step_jitted`, the
four-field volume-normalized after-state loss, reverse mode versus forward mode
and central finite differences in the registered direction, reverse/forward
relative error at most `1e-9`, reverse/FD relative error at most `0.05`, and
finite nonzero derivatives.  No physics, configuration, objective, direction,
step size, threshold, clipping, `stop_gradient`, or guard changes are in scope.

## Registered predictions and falsifiers

1. **One production step remains DEBT.** Prediction: reverse mode is non-finite
   while forward mode and central FD retain the round-2 finite values within
   the registered bars.  This is confirmed only if the unchanged row reports a
   reverse NaN and finite, mutually consistent forward/FD values.  A finite
   reverse derivative, a forward/FD failure, a changed primal, or a different
   failure mode refutes the prediction.
2. **The remaining owner is outside the landed TKE dissipation guard.** Round 2
   proved that guarding both inactive operands in
   `vertical_mixing/tke.py` leaves the one-step row unchanged.  Prediction: a
   statement-level VJP bisect through the executed production path finds a
   smaller sub-step closure whose primal is finite and whose transpose first
   emits a non-finite cotangent.  It is confirmed only by a finite downstream
   cotangent, a non-finite upstream cotangent, a cell census, and an executed
   path tied to the corresponding compiled NEMO statement.  Failure to isolate
   all four items leaves ownership **UNMEASURED**.
3. **Six CPU steps remain unavailable in this toolchain.** Prediction: the
   unchanged six-step CPU reverse/forward compilation reproduces round 2's
   `LLVM ERROR: Unable to allocate section memory!`, the known XLA-CPU
   large-function compilation failure.  The CI test must not start that
   compilation; it is an explicit skip carrying the exact reason.  A completed
   six-step row, a different abort, or successful stage-rematerialized CPU
   compilation refutes this prediction.
4. **GPU rows require XLA transcendentals.** Prediction: the operator runner
   reaches the card only under the explicitly recorded
   `PrecisionPolicy.fp64(transcendentals="native")`; libm must refuse on GPU at
   `core/transcendentals.py`.  Any GPU JSON lacking that exact policy in
   provenance is invalid.  Agreement is self-consistency under XLA
   transcendentals, not a certified bit-exact CPU/libm row.
5. **The gate remains fail-closed.** The `gyre_nemo_card` CPU parametrization
   must run one step only and pass solely by asserting the registered DEBT
   state, including the localized owner in its failure message.  Six steps are
   skipped with the LLVM reason.  A future finite reverse row must fail until a
   visible expectation edit promotes it to CONFIRMED.  The existing broken-VJP
   plant must continue to turn a passing synthetic control into REFUTED.

## Resource rows

CPU and GPU memory/wall-time rows remain **UNMEASURED** until a reverse pass
completes.  No peak from a failed compile and no unrelated earlier twin run may
be substituted.  Consequently every 360-step/60-day extrapolation remains
UNMEASURED unless a six-step peak exists for that backend and schedule.

