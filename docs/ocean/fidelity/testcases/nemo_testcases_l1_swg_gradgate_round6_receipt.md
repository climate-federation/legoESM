# Certified SWG reverse-mode gradient gate — round 6 receipt

Date: 2026-10-12

Status: **MEASUREMENT LANDED; PROMOTION HELD**. The requested affine timing
gate is executable and every required control fires, but all three fresh real
CPU suites are **REFUTED** by the preregistered contract. Nothing may train on
this card. This round changes no physics, card input, loss, random direction,
agreement bar, checkpoint schedule, configuration, stabilizer, clip, guard,
or `stop_gradient`.

## Preregistration and timing noise

Commit `0bc492a1d` preregistered the noise protocol and gate formula before any
round-6 measurement. Three separate clean processes measured the 360-step
`uniform` warm reverse at `4.091811742633581`, `4.501838743686676`, and
`4.092648733407259` seconds. Relative spread is defined by the repository
convention `(max - min) / abs(mean)` and measured
`0.09696137398733883`. The tolerance was therefore frozen before the
five-window rerun as
`max(0.05, 3 * spread) = 0.29088412196201646`; the floor did not bind.

Commit `ad02afb13` then encoded the identical gate for `none`, `uniform`, and
`binomial`. For an ordinary least-squares fit `time = a + b * steps`, it
requires:

1. `b > 0`, because more steps must have positive fitted marginal cost.
2. Strict monotonicity at all five windows.
3. `a >= 0`, because fixed overhead cannot be negative.
4. Every `abs(observed - fitted) <= tolerance * observed`, because each window,
   not only a summary score, must follow the affine model.
5. Every adjacent incremental cost lies within
   `[1 - 2*tolerance, 1 + 2*tolerance] * b`, so no interval can carry zero or
   anomalous marginal work while the global slope stays positive.

The residual test establishes linearity; the incremental test excludes a dead
interval; the intercept test excludes negative fixed overhead. No bound moved
after seeing a five-window result.

## Fresh-process CPU measurements

All fifteen rows start at the admitted kt=2 stage-1 entry under
`PrecisionPolicy.fp64(transcendentals="libm")`. Each row runs in its own clean
process. Time brackets only blocked execution of the already-jitted reverse
transform. Peak RSS is total process footprint, not an isolated adjoint
increment.

| steps | none time s / peak MiB | uniform time s / peak MiB | binomial time s / peak MiB |
|---:|---:|---:|---:|
| 24 | `0.455128 / 2763.473` | `0.337564 / 1704.301` | `1.090005 / 1903.852` |
| 96 | `2.551273 / 4002.828` | `1.299874 / 1735.891` | `3.882679 / 1916.066` |
| 360 | `7.388123 / 8206.043` | `4.523753 / 1822.562` | `15.333181 / 1849.938` |
| 720 | `12.982333 / 14079.523` | `8.276260 / 1943.598` | `29.736209 / 1872.863` |
| 1440 | `27.969813 / 25698.332` | `16.483339 / 2178.230` | `65.020948 / 2010.246` |

Every individual row is measurable, finite, nonzero, and CONFIRMED by the
unchanged resource-row classifier.

## Affine scaling verdict: retained refutation

| schedule | fitted a s | fitted b s/step | R2 | relative residual at 24 | failed checks | verdict |
|---|---:|---:|---:|---:|---|---|
| none | `0.230871` | `0.01901224` | `0.997100` | `0.509830` | residual | **REFUTED** |
| uniform | `0.211147` | `0.01131252` | `0.999501` | `0.429797` | residual | **REFUTED** |
| binomial | `-0.755118` | `0.04501463` | `0.998050` | `0.701623` | residual, intercept | **REFUTED** |

All three suites have positive slopes, strict monotonicity, and every
incremental cost inside the frozen band. Nevertheless, each 24-step residual
exceeds the measured `0.290884` tolerance; binomial also has a negative fitted
fixed overhead. High R2 does not rescue a failed per-window contract. The
prediction that every real suite would pass is retained as **REFUTED**. No bar
is changed to fit the observations.

The current executable also reclassifies all three Round-5 five-window suites
as **REFUTED**. Their earlier scaling verdicts are therefore **RETRACTED**;
their recorded times and memory peaks remain observations. The Round-6 JSONs
`round5_cpu_scaling_*_reclassified.json` make this retraction executable.

## Controls

All six required synthetic series return **REFUTED**, exit 1:

- constant time (`R2 = 0`);
- `[1, 100, 1, 1000, 2000]`;
- a decreasing endpoint;
- the reviewer's exact monotone series
  `[1, 6, 11.2500000225, 11.250000045, 16.87500009]`
  (`R2 = 0.8152955711`);
- one dead interval; and
- a constructed monotone series with `R2 = 0.9500000001`.

The last two controls fail the incremental-cost check; the R2 control also
fails the residual check. Thus neither a positive global slope nor a visually
high R2 can hide a dead or strongly curved interval.

## Retained directional agreement and direct 60-day cost

The native-policy GPU agreement rows are unchanged and remain CONFIRMED:

| steps | reverse/forward relative error | reverse/central-FD relative error |
|---:|---:|---:|
| 1 | `2.27064171926424e-16` | `3.57798841572499e-7` |
| 24 | `9.30692734647596e-16` | `2.1980290555407674e-6` |
| 720 | `2.3070259307585205e-16` | `9.041792746430404e-7` |
| 1440 | `3.02727997319582e-15` | `6.852553986852776e-6` |

The broken-VJP plant remains REFUTED. Agreement proves the three derivative
methods agree; it does not override the separate failed CPU scaling contract.

| schedule | CPU warm s / peak host MiB | GPU warm s / device peak growth MiB |
|---|---:|---:|
| none | `27.969813 / 25698.332` | `4.219477 / 15483.843` |
| uniform | `16.483339 / 2178.230` | `2.452661 / 510.466` |
| binomial | `65.020948 / 2010.246` | `6.252818 / 366.080` |

The direct 60-day uniform CPU observation is therefore `16.483339 s` and
`2178.230 MiB` total peak host RSS. The corresponding non-certified native
GPU observation is `2.452661 s` and `510.466 MiB` device peak growth. The
committed operator runner remains
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradgate_gpu_rows.sh`.

## Training composition and exact JSON fields

Training composition remains the measured per-step Python loop through the
production jitted `swe_rk3` step. The rejected uniform scan composition is not
used. Clean artifact `scan_composition_gap_24step.json`, produced at
`31b4e96cc`, now records semantic fields separately: SSH
`8.881784197001252e-16`, U `1.3877787807814457e-17`, and V
`6.071532165918825e-18`. The bitwise composition prediction remains
**REFUTED**; this last-bit gap is retained and not closed.

## Evidence and hygiene

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swggg_rounds/round6/`.

- The shared gate direct suite passes **35/35**; all six scaling plants exit 1.
- The isolated SWG production-step gradient-matrix test passes **1/1**.
- Independent, given-stage, chained-stage, carried-state, and 30-day SWG
  ladders remain AT-BAR and bit-identical; the stage one-ULP plant exits 1.
- Fresh fp64 hashes remain GYRE `f02d2f42a570f413`, TSUNAMI
  `8bb5ed6c1bc031bd`, VORTEX `f7489467bc156303`, and VORTEX_VEC
  `5041b91c67bd1443`.
- The VORTEX-card module reports only the inherited three digest-pin failures,
  with **3 failed, 69 passed**; no pin is changed here.
- The citation gate is run explicitly on this receipt; its shifted-citation
  plant exits 1.

## OPEN

- Binding lane result: all three real affine CPU suites are REFUTED, so this
  card remains excluded from training. Changing the bar is outside this
  measurement lane.
- GYRE six-step and GPU rows belong to `fidelity/gyre-gradient-gate`, not this
  lane.
- The scan composition's last-bit gap is retained and not closed.
- The deck-parameterised card lives on `fidelity/swg-double-gyre`, not this
  branch.
- The inherited stale GYRE/LOCK/OVERFLOW digest pins remain GYRE-lane debt.
  Expected values are `5850bbcfcd351c29` / `4ec82201f0fae7bf` /
  `7f1ec89b3e6578f5`; actual values are `f02d2f42a570f413` /
  `d5dfdf83b3a1e0ab` / `83de0010e1170032`. They are not re-pinned here.

No additional acquisition or work is scoped in this lane.

## Source citations

The certified compiled SWG dispatcher is the three-stage program at
`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:122-321`; its executed stage-3 momentum
block contains vector advection, ENS vorticity, constant Laplacian viscosity,
centered wind, and linear drag at
`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:260-273`. The total-ENS operator and its
live F-thickness divisions are in
`SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:639-675`.
