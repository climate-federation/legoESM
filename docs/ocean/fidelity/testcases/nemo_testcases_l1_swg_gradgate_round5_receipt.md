# Certified SWG reverse-mode gradient gate — round 5 receipt

Date: 2026-10-12

Status: **LANDED; MEASUREMENT SCOPE COMPLETE**. This round tightens only
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
pass before the lane can be complete. The preregistration above was committed
as `6da5d40988` before any round-5 resource row was run.

## Fresh-process CPU result

The tightened shared gate was committed as clean producer `cc594f691b`. Each
of the fifteen rows ran in a separate process from the admitted kt=2 stage-1
entry under `PrecisionPolicy.fp64(transcendentals="libm")`. The timer brackets
only blocked execution of the already-jitted reverse transform. Absolute peak
RSS is total process footprint, not an isolated adjoint increment.

| steps | none time (s) / peak MiB | uniform time (s) / peak MiB | binomial time (s) / peak MiB |
|---:|---:|---:|---:|
| 24 | `0.490248 / 2785.250` | `0.274183 / 1703.773` | `0.818509 / 1904.961` |
| 96 | `1.990220 / 3927.512` | `1.128513 / 1726.098` | `3.904149 / 1924.066` |
| 360 | `7.110497 / 8210.855` | `4.369059 / 1820.633` | `16.375659 / 1924.789` |
| 720 | `12.492354 / 14027.133` | `13.105090 / 1935.414` | `29.454459 / 1974.156` |
| 1440 | `25.376902 / 25715.344` | `21.576752 / 2175.957` | `60.555435 / 1948.523` |

All rows have finite nonzero losses and gradients, satisfy the unchanged
schedule-primal contract, and are individually CONFIRMED. The direct 60-day
uniform CPU gradient costs `21.576752 s` warm execution with `2175.957 MiB`
total peak host RSS in this rerun.

## Tightened scaling verdict and controls

| schedule | normalized ratios, 24-96 / 96-360 / 360-720 / 720-1440 | fitted s/step | fit R2 | verdict |
|---|---:|---:|---:|---|
| none | `1.014906 / 0.952725 / 0.878445 / 1.015697` | `0.0173767` | `0.998855` | **CONFIRMED** |
| uniform | `1.028978 / 1.032405 / 1.499761 / 0.823220` | `0.0155775` | `0.982177` | **CONFIRMED** |
| binomial | `1.192457 / 1.118513 / 0.899337 / 1.027950` | `0.0418936` | `0.999053` | **CONFIRMED** |

Every schedule is strictly increasing across all five windows and every
normalized adjacent ratio is inside the frozen `[0.5, 1.5]` band. The limiting
real observation is uniform 360-720 at `1.499761`; it passes without moving the
preregistered bound.

The same executable classifier returns **REFUTED**, exit 1, for all four
plants: constant time; the reviewer's `[1, 100, 1, 1000, 2000]` series; a
decreasing 1440-step endpoint; and the strictly increasing near-linear series
`[0.24, 0.96, 5.41, 7.2, 14.4]`, whose 96-360 normalized ratio is
`1.502778`. The last plant keeps every window increasing and keeps the retained
uniform 1440/720 ratio at `2.0`, so only the new adjacent-ratio check fires.
The first version of this plant, `[0.24, 0.96, 3.6, 10.81, 14.4]`, also failed
the retained uniform endpoint check and therefore did not isolate the new
contract; that control is **RETRACTED** and its artifact is retained as
`cpu_scaling_uniform_adjacent_ratio_outside_band_nondiscriminating_retracted.json`.
An additional strict-monotonicity isolation control
`[0.24, 0.96, 3.6, 3.6, 7.2]` keeps all four normalized ratios inside the
band (the equality interval is exactly the inclusive lower bound `0.5`) and
keeps every retained check passing; only strict monotonicity fails, so the
suite returns **REFUTED**, exit 1.

## Retained agreement, GPU cost, and composition

No agreement or GPU measurement changed in this round. The operator's
native-policy 1/24/720/1440-step reverse-forward relative errors remain
`2.27064e-16 / 9.30693e-16 / 2.30703e-16 / 3.02728e-15`; reverse-central-FD
relative errors remain `3.57799e-7 / 2.19803e-6 / 9.04179e-7 / 6.85255e-6`.
All four agreement rows are CONFIRMED.

At 1440 steps, native-policy GPU warm time / device peak growth remains: none
`4.219477 s / 15483.843 MiB`, uniform `2.452661 s / 510.466 MiB`, and binomial
`6.252818 s / 366.080 MiB`. The committed operator script is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradgate_gpu_rows.sh`.

Training uses the measured per-step Python loop through the production jitted
`swe_rk3` step. The retained 24-step uniform-scan composition differs from that
production loop by `8.881784197001252e-16` in SSH,
`1.3877787807814457e-17` in U, and `6.071532165918825e-18` in V; its bitwise
prediction remains REFUTED.

## Evidence and controls

Artifacts are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swggg_rounds/round5/`.
The fifteen `cpu_resources_*` JSONs, three `cpu_scaling_*` JSONs, and four
`*_plant.json` files stamp clean producer `cc594f691b`.

- The shared gate direct tests pass **32/32**.
- All three real scaling suites exit 0 CONFIRMED; all four required scaling
  plants and the added monotonicity-isolation control exit 1 REFUTED.
- The independent, given-stage, chained-stage, carried-state, and 30-day SWG
  ladders remain AT-BAR and bit-identical; the stage one-ULP plant exits 1.
- Under an explicitly instantiated fp64 policy, fresh card hashes remain GYRE
  `f02d2f42a570f413`, TSUNAMI `8bb5ed6c1bc031bd`, VORTEX
  `f7489467bc156303`, and VORTEX_VEC `5041b91c67bd1443`.
- A preliminary hash probe that enabled JAX x64 but omitted legoESM's fp64
  policy produced GYRE `3b2b358145ced147`; that artifact is retained as
  `card_digests_fp32_invalid_retracted.json` and the value is **RETRACTED**.
- The citation gate is run explicitly on this receipt; its shifted-citation
  plant exits 1.

## OPEN

The inherited stale GYRE/LOCK/OVERFLOW digest pins remain GYRE-lane debt.
Expected values are `5850bbcfcd351c29` / `4ec82201f0fae7bf` /
`7f1ec89b3e6578f5`; actual round-6 values are `f02d2f42a570f413` /
`d5dfdf83b3a1e0ab` / `83de0010e1170032`. They are not re-pinned here. No SWG
gradient measurement, operator acquisition, or gate-policy decision remains
open.

## Source citations

The certified compiled SWG dispatcher is the three-stage program at
`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:122-321`; its executed stage-3 momentum
block contains vector advection, ENS vorticity, constant Laplacian viscosity,
centered wind, and linear drag at
`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:260-273`. The total-ENS operator and its
live F-thickness divisions are in
`SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:639-675`.
