# Certified SWG reverse-mode gradient gate — round 6 receipt

Date: 2026-10-12

Status: **PREREGISTERED; MEASUREMENT NOT YET RUN**. This round replaces the
round-5 adjacent time-ratio classifier with the operator-specified affine
cost-model gate. It changes no physics, card input, loss, random direction,
agreement bar, checkpoint schedule, configuration, stabilizer, clip, guard,
or `stop_gradient`.

## Preregistered timing-noise measurement

Before changing the scaling classifier or rerunning any five-window suite,
run three fresh processes for the 360-step `uniform` CPU resource row. Each
uses the admitted kt=2 stage-1 entry under
`PrecisionPolicy.fp64(transcendentals="libm")` and times only blocked execution
of the already-jitted reverse transform. Define relative spread by the
repository convention `(max(time) - min(time)) / abs(mean(time))`. Freeze the
round-6 tolerance as `max(0.05, 3 * relative_spread)`; do not revise it after
seeing the five-window results.

## Preregistered affine cost-model gate

For each schedule independently, fit `time = a + b * steps` by ordinary least
squares over the frozen CPU windows `(24, 96, 360, 720, 1440)`. Require all of:

1. `b > 0`: more model steps must have positive fitted marginal cost.
2. Strict monotonicity at every adjacent window.
3. `a >= 0`: fixed overhead cannot be negative.
4. Every residual satisfies
   `abs(time_i - (a + b * steps_i)) <= tolerance * time_i`: each observed
   window must be consistent with the affine model, not merely raise a summary
   fit statistic.
5. Every consecutive incremental cost
   `(time[i+1] - time[i]) / (steps[i+1] - steps[i])` lies in
   `[1 - 2*tolerance, 1 + 2*tolerance] * b`: no interval may carry zero or an
   anomalous marginal cost while the global fit remains positive.

The residual test establishes linearity; the incremental-cost test rejects a
dead interval; the intercept test rejects a physically impossible negative
fixed overhead beyond measured noise. The identical measured tolerance and
checks apply to `none`, `uniform`, and `binomial`.

The executable controls must REFUTE: constant time;
`[1, 100, 1, 1000, 2000]`; a decreasing endpoint; the Round-5 reviewer's exact
series `[1, 6, 11.2500000225, 11.250000045, 16.87500009]`; a series with one
dead interval; and a series with `R2 = 0.95`. All three fresh real suites must
pass. The exact SSH, U, and V production-loop versus scan-composition gaps must
also be written as separate JSON fields.

The three preregistered 360-step uniform warm times, acquired from separate
clean processes at producer `0bc492a1d`, are `4.091811742633581`,
`4.501838743686676`, and `4.092648733407259` seconds. Their relative spread is
`0.09696137398733883`; therefore the frozen tolerance is
`0.29088412196201646`. The `0.05` floor does not bind. These are the only
measurements made before freezing the gate.

## OPEN

Pending the gate implementation, controls, five-window rerun, ladders, card
digests, and explicit receipt citation gate.
