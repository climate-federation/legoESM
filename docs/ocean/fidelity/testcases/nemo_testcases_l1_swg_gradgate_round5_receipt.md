# Certified SWG reverse-mode gradient gate — round 5 receipt

Date: 2026-10-12

Status: **PREREGISTERED; MEASUREMENTS NOT YET RUN**. This round tightens only
the executable wall-time scaling classifier requested after the round-4
review. No physics, card input, loss, random direction, agreement bar,
checkpoint schedule, configuration, stabilizer, clip, guard, or
`stop_gradient` changes.

## Preregistered scaling contract

The same contract applies to `none`, `uniform`, and `binomial` over the frozen
CPU windows `(24, 96, 360, 720, 1440)`:

1. Warm reverse time must increase strictly at every adjacent window.
2. For every adjacent pair `(lower, upper)`, the normalized ratio
   `(time[upper] / time[lower]) / (upper / lower)` must lie in the inclusive
   band `[0.5, 1.5]`.
3. The retained positive fitted slope and direct 1440-over-720 checks remain.
   The retained uniform 1440/720 absolute ratio must remain in `[1.5, 2.5]`.

Linear per-step adjoint work predicts a normalized adjacent ratio of `1.0`.
The broad `0.5..1.5` band allows a 50% departure for fixed launch overhead and
host timing noise while still requiring every interval to carry work
commensurate with its additional steps. The already-recorded round-3 rows,
used to design rather than evaluate this round, span `0.929..1.326` across all
twelve normalized adjacent ratios. The band and the inclusive endpoints are
frozen by this preregistration before any round-5 row is rerun.

The executable controls must REFUTE: constant times; the reviewer's
`[1, 100, 1, 1000, 2000]` seconds; a decreasing final endpoint; and an
otherwise near-linear, strictly increasing series with at least one normalized
adjacent ratio just outside the frozen band. All three fresh real suites must
pass before the lane can be complete.

## OPEN

- Implement the frozen checks and all four plants in the shared gate tests.
- Rerun all fifteen CPU resource rows in fresh processes, then classify the
  three five-window suites.
- Amend the round-4 receipt without changing its recorded measurements.
- Run the direct tests, explicit citation gate and shifted-citation plant, and
  commit the final receipt.
