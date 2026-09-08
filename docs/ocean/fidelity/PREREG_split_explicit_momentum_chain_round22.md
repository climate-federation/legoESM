# Preregistration amendment: row-1.3 EEN coefficients, round 22

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Frozen before the held NEMO build
and run. This amendment authorizes instrumentation only, not a physics fix.

Round 21 clears the surface-pressure-gradient operand exactly on 9,758 U and
9,868 V points. The next executed operand is the pure live EEN Coriolis call
at `dynspg_ts.F90:783-805`. NEMO first freezes its eight coefficient arrays in
`dyn_cor_2D_init` (`dynspg_ts.F90:1491-1667`) and then applies four coefficients
per component in source association order (`dynspg_ts.F90:1670-1693`).

The held ON arm writes `ffu_{nw,ne,sw,se}` and `ffv_{nw,ne,sw,se}` immediately
after the first live call. Its OFF arm carries the deterministic 197-stream
stack plus the already-certified six QCO streams. The bracket must therefore
show exactly 203 shared byte-identical streams and exactly eight new 90,944-byte
coefficient streams. A one-bit shared-stream plant and a missing-new-stream
plant must both fail the gate.

The CPU scorer first reconstructs `zu_trd/zv_trd` from the dumped coefficients,
dumped `ua_e/va_e`, and the literal parentheses at `dynspg_ts.F90:1682-1691`.
The pointwise bar is normalized RMS and maximum error/NEMO RMS both at most
`1e-15`; exact bits are reported separately. Failure leaves the writer or
application transcription invalid and stops the lane.

If reconstruction is at bar, four checkerboard linear-response probes per
component materialize the production EEN operator's effective NW/NE/SW/SE
coefficient arrays. Each of the eight arrays is compared on the registered
wet U or V population with the same `1e-15` pointwise bar. A coefficient is
owned only if its effective production array is DEBT while the literal NEMO
reconstruction remains at bar. If all coefficient arrays are at bar but the
production output is DEBT, the remaining owner is multiplication/addition
association in the live application. Mixed failures are reported as a
coefficient composition and no single-owner physics change is authorized.

Rows 1.4, 2--6, the free-surface filter, momentum RHS, and tracer tail remain
ordered-blocked until this first row-1.3 operand is disposed.
