# SWG online-learning design

Date: 2026-10-13

Status: **phase-3 harness built; optimization held**. The closure seam, network,
balanced loss, segment runner, optimizer construction, profiles, and evaluation
gates exist and are tested. No optimization update is authorized until the
balanced-start acquisition, stationarity assessment, 3.125 km
reference-adequacy gate, and coarse-composition decision close. The learned
term closes only the resolution gap. It may not compensate a legoESM/NEMO gap.

## Fixed physical model and target convention

The training model is the NEMO-certified `dg50` card: 50 km, 400 s,
500 m, full gravity `g=9.80665 m s-2`, and an explicit free surface. The
per-axis gravity-wave Courant number is 0.560190, and the two-dimensional RK3
imaginary argument is 1.58446, below `sqrt(3)`. The training model remains
`nu=10 m2 s-1`; `nu=80 m2 s-1` is a separate grid-Reynolds control, never a
silent replacement.

The stage ladder remains the exact certification authority. A separate 30-day
composition audit found that none of the actual execution compositions is
bit-identical to NEMO. Per-step has 249/690 exact rows, 561/690 within
`1e-15`, and maximum error `9.479e-15`; compiled-day and flat-segment each have
249/690 exact, 543/690 within `1e-15`, and maximum error `5.218e-15`. All first
differ on day 1 at `vtau_before[60,1]` by signed zero (one ordered IEEE word),
then first exceed the numerical bar on day 9 at wall-adjacent `sshn[1,1]`
(16 ULP per-step; 25 ULP compiled/flat). Therefore this document names the
training composition but does not call it NEMO bit-for-bit; training remains
held for the user's explicit approximately-`1e-14` composition ruling.

The state target is the conservative restriction in `swg_coarse_graining.py`:
sum fine T-cell volumes into each coarse T cell and sum `e2*h*u` or `e1*h*v`
over the fine faces tiling each coarse face, then diagnose coarse SSH and face
velocity using the coarse live QCO thickness. There is no component-mean
restoration. The real day-1095 continuity residual is `4.520e-18 m s-1`
interior and `2.887e-18 m s-1` in the outer four cells. The round-1
independent-state Gaussian remains a labelled diagnostic and is forbidden as a
training target.

Gaussian smoothing is optional only for the derived, already restricted
relative-vorticity target. It uses reflecting closed walls and `sigma=1`
coarse cell; `sigma` is the standard deviation, not an unspecified “width.”
This matches Maddison's convention, which says “width (specifically a standard
deviation) of 62.5 km,” while changing the state restriction to the
continuity-conserving operator required by this free-surface C grid. Maddison
filtered instantaneous relative vorticity, then subsampled; he did not
independently filter face velocities. See
[Maddison (2026), sections 2.3 and 2.4](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004883).

The reference is not yet adequate or stationary. At 6.25 km the Munk width has
1.27 cells; restricted daily-mean truth gives consecutive annual mean-velocity
differences of 0.04765 and 0.04203 m s-1 for years 1-to-2 and 2-to-3. A 3.125 km
sensitivity is therefore mandatory. On the common 50 km grid, each common year
must agree within 10% in basin KE, basin EKE, jet-position RMSE divided by basin
width, and KE-spectrum relative L1 error. Any failed yearly score makes
reference adequacy a user decision. Train/validation/test dates are frozen only
after adequate, stationary years exist; splits are chronological and no start
window may cross a split boundary.

## Initialization and hourly observations

Ten starts are fixed at fine days 731, 771, ..., 1091. Each supplies a four-day
hourly segment restricted inside the compiled fine-model scan, retaining only
coarse T/U/V arrays. A restarted segment must reproduce the uninterrupted next
fine daily state bit-for-bit under the same fp64/native policy. Maddison's loss
has “`M=24w` hourly outputs”; this lane likewise scores 24, 48, 96, or 192
right-endpoint observations for 1, 2, 4, or 8 days.

Before training, every start receives paired 48-hour no-network dg50 forecasts
from the old raw-filtered state and the conservative state. At hours 6, 24, and
48 the deciding statistic is

`RMS(zeta_raw - zeta_conservative) / RMS(zeta_conservative - zeta_target)`.

The instrument also reports divergent kinetic energy, hourly SSH-tendency RMS,
and vorticity error. The conservative start is admitted only if this ratio is
at most 0.1 at all three leads for at least 8 of 10 starts and it does not have
larger median divergent energy than the raw start. Otherwise training remains
blocked while a rotational/geostrophic adjustment is designed and measured.
Raw-filtered initialization is never an allowed fallback.

## Closure seam and pytree contract

The closure is an Equinox module producing one scalar potential on native
F points. The card applies the matched rotated gradient to form native U/V
accelerations at **every RK3 stage**. The same F-point potential is constant on
the closed boundary, so wall-normal force is zero; the discrete divergence of
the volume-force transport is roundoff zero. The seam is a pure pytree term
inside the certified step. If the closure argument is absent, no addition or
reassociation occurs, preserving the certified card byte-for-byte.

Inputs are coarse SSH, native U/V velocity averaged to T, wet mask, normalized
`e2u/e1v` metrics, an explicit RNG key, and a fixed-shape noise pytree. The
phase-3 Equinox model has five width-3 convolutions, 16 hidden channels, ELU,
an 11-cell (550 km) receptive field, and 7,984 parameters. Its output is the
native F-point potential; U/V acceleration is derived, never independently
predicted. The final convolution has no bias because a constant-potential bias
would be annihilated by the curl and become inert. Noise enters the SSH channel
through the same state weights, so deterministic zero noise leaves no
noise-only trainable leaf. This is intentionally smaller than Maddison's seven
width-5, 32-filter residual convolutions with 154,857 weights (paper lines
436--442). Architecture and normalization scales are mandatory manifest data,
not training defaults.

The closure module, noise array, noise scale, and RNG key are explicit traced
arguments. Per the round-3 ruling, the analytically steady wind is closed over
by the segment builder (it is neither time-varying nor trainable). Separate
functions are built once for each admitted window. Replacing parameters,
noise, or keys at fixed shapes does not retrace.

Before an optimizer update exists, synthetic tests must fail on planted violations of
F-to-U/V staggering, zero wall-normal force, divergence, integrated momentum,
absence-is-exactly-zero, and JVP/finite-difference agreement. Static masks and
metrics are not trainable. Every trainable leaf must have a finite nonzero loss
gradient under the repository's no-inert-parameter gate. The gate runs on the
first-step seam and on every reverse profile; one-step and 24-hour JVPs agree
with centred finite differences, and closure-free 4/8-day segment outputs are
bit-identical at their 96 common hourly endpoints.

## Loss, rollout, and curriculum

The deterministic point loss is swappable and ensemble-aware from its first
version. Given `(members, hours, fields)`, the point-loss implementation uses
one member and the energy-score implementation requires `S >= 2`. The balanced
point loss is the equal mean of training-split-variance-normalized, wet-area
L2 errors in Gaussian-smoothed relative vorticity and the two components of the
Dirichlet-streamfunction rotational velocity. Raw SSH is a monitored fast-mode
diagnostic, not a fitted field. Hourly weights are uniform, and spatial weights
are native wet areas. Normalizers are derived only from the training split and
content-pinned in the run manifest.

The first curriculum is exactly 1, 2, 4, and 8 days. A selected checkpoint
initializes the next window. No 16, 32, or 60-day deterministic stage is
authorized until the real network, batch, loss, and optimizer have been
profiled. For scale, 60 days is 12,960 coarse steps and projects to about
22.1 s and 4.49 GiB per sample under uniform checkpointing, beyond the measured
10.4248-day provisional e-folding time. Uniform production-step checkpointing
is the default; alternatives require a measured memory/time receipt.

Optimizer construction must use `legoesm.ml.training.create_optimizer`.
Learning rate, warmup, batch size, start sampler, split dates, architecture,
and parameter count are explicit manifest fields, not defaults. The
truth-generation day scan and training segment loop are distinct named
compositions. With zero closure, 4-day and 8-day runs from the same coarse state
must match bit-for-bit in native T/U/V at every common hourly endpoint. A
failure blocks training rather than widening a tolerance.

The phase-3 CPU profile used the real 7,984-parameter network, one sample, and
uniform production-step rematerialization. These are execution-only profiles;
the temporary unit-valued field scales are not admitted training
normalizations and no optimizer update was taken.

| Window | Steps / observations | Forward execution / peak RSS | Reverse execution / peak RSS |
|---:|---:|---:|---:|
| 1 day | 216 / 24 | 2.15 s / 1.16 GiB | 7.20 s / 1.96 GiB |
| 2 days | 432 / 48 | 4.45 s / 1.24 GiB | 14.35 s / 2.13 GiB |
| 4 days | 864 / 96 | 8.56 s / 1.39 GiB | 28.36 s / 2.50 GiB |
| 8 days | 1,728 / 192 | 17.45 s / 1.69 GiB | 58.87 s / 3.23 GiB |

## Controls and held-out gates

Both no-closure controls (`nu=10` Maddison replication and `nu=80`
grid-Reynolds control) use the same evaluation harness. The `nu=80` control
requires a finite, positive-thickness, sub-unit-Courant 30-day preflight before
its 20-year CPU run. Reports are yearly, map scores use wet cells only, and
health includes radial top-octave KE/enstrophy fractions and a native
checkerboard index. Years 2–3 are currently the least-confounded exploratory
window, not an equilibrated claim.

Each trained window has at least three network seeds and a 20-year free run.
Every seed must remain finite with positive thickness and `Cu,Cv < 1`. On the
held-out years, the median seed must beat the **better** of the two controls in
mean-transport absolute error, KE-spectrum relative L1 error, and wet-cell EKE
map normalized RMSE. Every seed must retain basin EKE within 20% of filtered
truth and must not exceed the worse control's top-octave fractions or
checkerboard index. Confidence intervals and raw seed scores are reported; a
median alone cannot hide an unstable seed.

The executable scorecard first measured both controls on years 2--3. The
better-control thresholds are `38.309 m2 s-1` mean-transport RMSE,
`0.188445` KE-spectrum relative L1, and `0.334802` wet EKE-map NRMSE. `nu=10`
retains `0.73270` of truth EKE with top-octave KE/enstrophy fractions
`0.18178/0.75783`; `nu=80` retains `0.39659` with `0.15100/0.72942`. Thus a
network that merely removes grid noise or merely restores eddy energy cannot
pass all gates.

Two adversarial controls are mandatory. Replacing every time-varying anomaly
by its time mean must fail the EKE-retention gate (collapsed variance).
Adding a native two-cell checkerboard calibrated above the worse control must
fail spectral health. A gate release that lets either plant pass is itself a
failed gate. Mean transport merely lying between a baseline and truth is not
an acceptance criterion.

## Stochastic branch

The rollout API always carries an explicit RNG key and member-indexed noise
pytree. The loss API accepts `S` trajectories and dispatches between point loss
and energy score without changing solver or closure signatures. Energy-score
training requires `S >= 2`; member RNG streams are folded by step and member.
There is no inactive trainable “noise parameter” in deterministic mode. The
noise scale is an explicit non-parameter input; noise uses the same weights as
the state path. Any future choice to make its scale trainable must first pass
the no-inert-parameter gate. Collapsed ensemble variance is a planted failure.

## Review requirements and enforcement

| # | Required change | Design answer | Enforcing test or measurement |
|---|---|---|---|
| 1 | Conservative, balanced initialization | Exact volume/face-transport restriction; raw state forbidden | Real continuity audit plus ten-start 6/24/48 h shock ratio |
| 2 | Hourly targets | 24 right-endpoint restricted samples/day, no fine-state storage | Ten four-day caches; restart-next-day bitwise gate |
| 3 | Adequate stationary reference | Stationarity scan and 3.125 km sensitivity before splits | Four yearly <=10% adequacy scores; annual velocity-difference report |
| 4 | Balanced predictability | Smooth zero-divergence vorticity perturbation, two amplitudes, JVP | 120-day twins and registered balanced fit rule |
| 5 | Cap curriculum | Initial windows 1/2/4/8 days only | Network-inclusive profile required before 16/32/60 days |
| 6 | Non-divergent forcing | F-potential rotated gradient at every RK3 stage | Divergence, walls, momentum, zero seam, JVP/FD plants |
| 7 | Viscosity controls | `nu=10` and `nu=80` are separate named controls | 30-day preflight, then identical 20-year health/climate harness |
| 8 | Structural stochastic hook | Explicit RNG/noise and ensemble-aware swappable loss, `S>=2` | API shape tests, RNG independence, inert-noise and collapse plants |
| 9 | Quantitative multi-seed gates | Three seeds; beat best control; variability and noise bounds | Held-out harness plus collapsed-variance/grid-noise failures |
| 10 | Correct physics and composition | Full-gravity explicit free surface; named truth/training loops | Zero-closure 4/8-day bitwise A/B and manifest composition fields |

## Phase-3 implementation disposition

| Requirement | Design answer | Test or measurement |
|---|---|---|
| F-potential seam | Matched live-thickness transport curl at every RK3 stage; absent branch makes no addition | Random-potential divergence/wall/integral tests; unchanged dg50 ladder |
| Explicit stochastic structure | RNG, member noise, and scale are traced; shared weights avoid an inactive noise branch | RNG/no-retrace tests and no-inert gradient gate |
| Balanced target | Sigma-one smoothed vorticity plus Dirichlet-Poisson rotational U/V | Point/energy-score shared-interface tests; `S=1` energy score refused |
| Initial curriculum | Builders admit only 1/2/4/8 days | Refusal test for 16 days and measured per-window profiles |
| Segment composition | Flat hourly scan with uniform per-step rematerialization; steady wind closed over | 4/8-day zero-closure common-endpoint bit identity |
| Optimizer | Repository `create_optimizer`, initialized during profiles | Metadata/direct test; `training_performed=false` |
| Multi-seed gate | Median beats better control on three skill measures; every seed meets variability/noise bounds | Three-seed minimum plus failed collapsed and grid-noise plants |
| Coarse controls | `nu=10` and `nu=80` separately scored on years 2--3 | Content-pinned round-2 array hashes in round-3 scorecard |
| Model identity | Exact and numerical composition scores are kept separate | Three 30-day artifacts with first-cell ULP localization |
| Training release | Requires balanced start, adequate/stationary reference, and user composition ruling | Explicit BUILD-NOT-TRAIN hold; no optimization update in phase 3 |
