# Certified SWG reverse-mode gradient gate — round 4 receipt

Date: 2026-10-12

Status: **HELD FOR ONE BAR DECISION; MEASUREMENTS COMPLETE**. The historical wall-time
retraction is executable, the five-window CPU cost model has a binding suite
and constant-time plant, and all operator GPU resource rows are classified
without turning the preregistered schedule ordering into an acceptance gate.
No physics, card input, loss, random direction, agreement bar, checkpoint
schedule, configuration, stabilizer, clip, guard, or `stop_gradient` changed.

## Provenance and review blockers

The incoming tip was `1d7fae76681ecd6e551bebca6892a1953d2e9c79`. The round-4
instrument repair is `349eaad0e842951830636cbacbe1b435a96e62a1`. Classifier
artifacts, including the rejected review experiment at clean `fc1a36478`, are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swggg_rounds/round4/`
and stamp their producer commits. The final classifier retains only the
preregistered T1/T2 checks described below. The GPU measurements remain
operator-acquired artifacts from clean `1d7fae766`; they are classified, not
re-derived.

Both confirmed round-3 review blockers are closed. A resource row now returns
RETRACTED unless it carries the exact repaired compiled-transform and timer-
span stamps. Running the executable over the three round-2 1440-step files
returns RETRACTED for every row and for the suite. The round-1 and round-2
receipt tables now label their timing and increment columns RETRACTED while
retaining only the absolute peak-RSS observations.

The five-row CPU suite applies the preregistered round-3 predictions of positive
affine-fit slope and direct 1440-step time greater than 720-step time for every
schedule, plus the existing uniform 1440/720 ratio band `[1.5, 2.5]`. Round 4
also reports every adjacent interval's seconds-per-step divided by the fitted
slope, but does not classify those diagnostics: the round-3 preregistration
explicitly says fit residuals are descriptive and no new timing bar may be
introduced. All three measured suites return CONFIRMED. Replacing all five
times by the 24-step time makes every suite REFUTED, with zero slope and equal
720/1440 times.

## GPU resource prediction: retained refutations

Every individual native-policy GPU row is measurable: JAX allocator peak
growth and the secondary `nvidia-smi` increment are positive. The registered
`none > binomial > uniform` ordering is a prediction, not a gate for acquiring
later rows. The repaired runner records REFUTED, VACUOUS, or RETRACTED suites
and continues; only a missing or malformed result is an instrument refusal.

| steps | JAX increment MiB, none / uniform / binomial | `nvidia-smi` increment MiB, none / uniform / binomial | sampler ordering | prediction verdict |
|---:|---:|---:|---|---|
| 24 | `268.602 / 32.001 / 368.378` | `532 / 42 / 526` | JAX: binomial > none > uniform; secondary: none > binomial > uniform | **REFUTED** |
| 720 | `7751.822 / 256.001 / 368.378` | `8212 / 266 / 526` | both: none > binomial > uniform | **CONFIRMED** |
| 1440 | `15485.377 / 512.001 / 368.378` | `16404 / 522 / 526` | JAX: none > uniform > binomial; secondary: none > binomial > uniform | **REFUTED** |

The 24-step refutation is retained. A plausible, not measured, interpretation
is that binomial's checkpoint stack and recomputation scratch exceed the tape
for only 24 small residual sets. The 1440-step refutation is also retained;
the JAX allocator measures uniform above binomial even though the secondary
sampler separates them by only 4 MiB. No ordering bar was changed to fit either
result. The forced none measurement under the uniform label makes both sampler
orderings false in the 720-step CONFIRMED suite and returns REFUTED.

## Direct sixty-day CPU and GPU cost

CPU rows use the certified fp64/libm policy and report total process peak host
RSS. GPU rows use the non-certified fp64/native policy and report fresh-process
JAX device peak growth. Times are blocked warm executions of the pre-jitted
reverse. They are costs on different hardware/policies and are not a speedup
comparison.

| schedule | CPU warm time (s) | CPU peak host RSS (MiB) | GPU warm time (s) | GPU device peak growth (MiB) |
|---|---:|---:|---:|---:|
| none | `15.830778` | `25652.617` | `4.219477` | `15483.843` |
| uniform | `10.883964` | `2185.824` | `2.452661` | `510.466` |
| binomial | `45.503263` | `1956.293` | `6.252818` | `366.080` |

The true measured 60-day uniform CPU gradient cost is therefore
**10.883964 s** and **2185.824 MiB total peak host RSS** on the CPU host. The
corresponding native-policy GPU row is **2.452661 s** with **510.466 MiB JAX
device peak growth**.

## Directional agreement and training composition

No agreement calculation changed. The complete operator GPU agreement result
is:

| steps | reverse/forward relative | reverse/central-FD relative | verdict |
|---:|---:|---:|---|
| 1 | `2.27064171926424e-16` | `3.57798841572499e-07` | **CONFIRMED** |
| 24 | `9.30692734647596e-16` | `2.1980290555407674e-06` | **CONFIRMED** |
| 720 | `2.3070259307585205e-16` | `9.041792746430404e-07` | **CONFIRMED** |
| 1440 | `3.02727997319582e-15` | `6.852553986852776e-06` | **CONFIRMED** |

Training on this card uses the measured per-step Python loop: each step crosses
the production jitted `swe_rk3` boundary and is rematerialized as one step. It
does not use the retained uniform-scan composition. That scan's 24-step primal
differs from production by `8.881784197001252e-16` in SSH,
`1.3877787807814457e-17` in U, and `6.071532165918825e-18` in V. The bitwise
prediction remains REFUTED; the last-bit gap is not waived.

## Evidence and controls

The nine operator GPU input-row SHA-256 values, grouped as
`none/uniform/binomial`, are: 24 steps
`638f7c32d8105965b1557a29313a1614448c4b4d8d78c73b216e772b5a31b4da` /
`703274238b93534e6e08feb8ce956b44b94d0f28ef5b0ac1f13f25a35dafc51c` /
`55106ac0786bef6876fb7a085c358e3f0561fce3edd41006715018d31b9e62bc`;
720 steps
`250715280d1f0e5e2b5fbe0f174b3610370ebfbbf40b956bde930a6d0c0ecf0d` /
`38fd537509c7a7170ef8d8d137aa51f4db18eb0bc67698c5a8d1ed121b81e725` /
`3f3c433ec1e5092fa5f05a49c9b9ca90f5589951561154725bb3a2a9b632649e`;
and 1440 steps
`8e8a0026394a87feead294f788b0e654df2a6a53558a99685e7f873633576a3c` /
`561631137b4e0df79d8dae4887ad13abf2cbbf442a0340970536719d2d20dbc0` /
`b738acb56b7e8d95735ed43beb6a8bbd2db4520a46dc19759832a5049cea63bc`.

- The shared gate direct tests pass **28/28**, including executable legacy-row
  retraction, the cost-model suite and constant-time plant, allocator refusal,
  and one-sampler-per-pass GPU control flow.
- The isolated SWG production gradient-matrix row passes **1/1**.
- The GPU runner passes shell syntax checking; its executable self-test walks
  REFUTED, VACUOUS, RETRACTED, then CONFIRMED rows without premature exit.
- The independent, stage, carried-state, and 30-day SWG ladders remain AT-BAR
  and bit-identical; the stage one-ULP and broken-VJP plants exit 1.
- Fresh card hashes remain GYRE `f02d2f42a570f413`, TSUNAMI
  `8bb5ed6c1bc031bd`, VORTEX `f7489467bc156303`, and VORTEX_VEC
  `5041b91c67bd1443`; `round4/card_digests_final.json` identifies the
  committed digest probe and stamps clean producer commit `fc1a36478`.
- The citation gate is run explicitly on this receipt; its shifted-citation
  plant exits 1.

## OPEN

The inherited stale GYRE/LOCK/OVERFLOW digest pins remain GYRE-lane debt.
Expected values are `5850bbcfcd351c29` / `4ec82201f0fae7bf` /
`7f1ec89b3e6578f5`; actual round-6 values are `f02d2f42a570f413` /
`d5dfdf83b3a1e0ab` / `83de0010e1170032`. They are not re-pinned here. No SWG
gradient measurement or operator acquisition remains open. One gate-policy
decision remains: either keep the preregistered T1/T2 scaling verdict, or
authorize and preregister a numeric interval-to-fit acceptance band in a later
round. This receipt does not invent that threshold post hoc.

## Source citations

The certified compiled SWG dispatcher is the three-stage program at
`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:122-321`; its executed stage-3 momentum
block contains vector advection, ENS vorticity, constant Laplacian viscosity,
centered wind, and linear drag at
`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:260-273`. The total-ENS operator and its
live F-thickness divisions are in
`SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:639-675`.
