# Preregistration: split-explicit chain round 13 — production acceptance

Date: 2026-08-29. Status: **FROZEN BEFORE MEASUREMENT**.

Rounds 11--12 authorize the NEMO-order QCO continuity fix: both metric
transports are uniquely literal-owned, and the literal divergence is the only
bit-exact arm (0/9,920 mismatches versus 2,655 for divide-after-sum).

Instrument amendment before acceptance: round 11 said the "unchanged round-9
production scorer" would be rerun. That scorer manually reconstructs the now
retired generic `H*U -> divergence_cgrid` arithmetic, so it cannot measure the
new production path. No acceptance score was run with it. The committed
round-13 scorer instead calls the production functions
`nemo_ssh_avg_face_depth`, `nemo_literal_metric_transports`, and
`nemo_literal_continuity_divergence` directly on the same six dumps.

It re-emits ordered rows 9.1--9.7 under the unchanged gates. Acceptance
requires every row AT BAR: POINTWISE max/RMS <= `1e-15` for 9.1--9.5 and
ACCUMULATING max/RMS <= `1e-12` for 9.6--9.7, with the unchanged correlation
and mean-absolute-ratio axes. Identity and a `1e-6` RMS planted violation must
traverse the same classifier. All prior receipt, dump, runtime, session,
package, CPU, fp64, population, and clean-tree bindings remain mandatory.

If all seven rows clear, row 1.3 is closed and row 1.4 is released. Otherwise
the first DEBT remains the ordered stop. No NEMO run/build, GPU, `mpirun`, or
push is authorized.
