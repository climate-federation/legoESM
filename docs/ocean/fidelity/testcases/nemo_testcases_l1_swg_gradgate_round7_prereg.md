# Certified SWG reverse-mode gradient gate — round 7 preregistration

Date: 2026-10-12

Status: **PREREGISTERED BEFORE ROUND-7 MEASUREMENT**. This final round replaces
the repeatedly defeated wall-time linearity gate with a measurement carrying
observed uncertainty. It changes no physics, card input, loss, random
direction, agreement bar, checkpoint schedule, configuration, stabilizer,
clip, guard, or `stop_gradient`.

## Correction to the round-6 record

Round 6 did not implement an identical preregistered intercept rule. Commit
`0bc492a1d` preregistered `a >= -tolerance * mean(time)`; after the timing-noise
measurement, commit `ad02afb13` changed that rule to `a >= 0`. The stricter
rule was motivated by the interpretation of `a` as fixed overhead, but making
that change after measurement was an unauthorized **DEVIATION**. It did not
change the held outcome because every real schedule independently failed the
residual rule. Round 7 removes the intercept rule together with the other
uncertifiable affine-linearity rules; the fitted intercept remains descriptive.

## Frozen acquisition and uncertainty calculation

For each of `none`, `uniform`, and `binomial`, acquire three fresh-process CPU
rows at each frozen window `(24, 96, 360, 720, 1440)`. Every row starts from
the admitted kt=2 stage-1 entry under
`PrecisionPolicy.fp64(transcendentals="libm")`. The timer continues to bracket
only blocked execution of the already-jitted reverse transform. The classifier
requires repeat labels `1, 2, 3`, fifteen distinct process IDs per schedule,
and an individually measurable resource row for every input.

For each window, report all three times, their mean, sample standard deviation,
full range, and relative full spread. Fit `time = a + b * steps` to the five
window means. Because every window has three repeats, this is the same slope as
ordinary least squares over all fifteen observations. Report:

- `b` in seconds per step;
- one-standard-error uncertainty of `b`, computed from all fifteen residuals;
- within-window repeat residual SSE and RMS;
- replicated five-mean fit-residual SSE and five-mean fit-residual RMS; and
- the fitted intercept, R2, per-window residuals, and adjacent incremental
  costs as descriptive diagnostics only.

The slope standard error uses
`SSE_total = SSE_repeat + 3 * SSE_fit_means`, 13 residual degrees of freedom,
and the step denominator replicated three times. It is an uncertainty estimate,
not a confidence interval and not a linearity verdict.

The 60-day CPU deliverable will quote the 1440-step mean warm time plus or
minus one sample standard deviation, the full time range, and the maximum
observed total host-RSS peak across the three processes. GPU time and memory
remain the operator-acquired native-policy measurements at clean
`1d7fae766`; no GPU rerun is scoped.

## Executable classification and plants

The only retained acceptance condition is strict increase of the five window
mean times. Its reason is limited: a longer adjoint window cannot have negative
incremental work. It does not certify an affine law. A real suite satisfying
that condition is classified **MEASURED**, never CONFIRMED or linear; a suite
that fails it is REFUTED. Constant time, a decreasing endpoint, and the exact
nonmonotonic series `[1, 100, 1, 1000, 2000]` must therefore be REFUTED.

The former affine-gate controls are retained as diagnostics, not acceptance
tests. The exact review counterexample
`[0.2451153271, 1.4510041507, 4.6135624796, 6.2600150596, 16.3588091356]`,
the earlier monotone nonlinear series, the dead-interval series, and the
R2≈0.95 series must be emitted as **MEASURED** old-gate plants with their large
fit residuals visible. That status is a recorded measurement, not a pass and
not a claim of linearity.

Evidence will be written outside git under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swggg_rounds/round7/`.

## Frozen OPEN ledger

- GYRE six-step and GPU rows belong to `fidelity/gyre-gradient-gate`.
- The scan composition's last-bit gap is retained and not closed.
- The deck-parameterised card lives on `fidelity/swg-double-gyre`.
- The inherited stale GYRE/LOCK/OVERFLOW digest pins remain GYRE-lane debt.
- Wall-time linearity is not certifiable on this shared node.

## Source citations

The certified compiled SWG dispatcher is the three-stage program at
`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:122-321`; its executed stage-3 momentum
block contains vector advection, ENS vorticity, constant Laplacian viscosity,
centered wind, and linear drag at
`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:260-273`. The total-ENS operator and its
live F-thickness divisions are in
`SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:639-675`.
