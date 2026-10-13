# SWG online-learning design

Date: 2026-10-13

Status: **Round-5 pre-training implementation complete; acquisitions held**.
One real CPU optimizer step remains a seam test, not training. The user has
accepted the recorded dg50 composition debt, but the dual-operator start
experiment, the content-pinned multi-year hourly data, and the 3.125 km
adequacy result remain prerequisites to the first training run. The learned
term closes only the resolution gap and may not compensate a legoESM/NEMO gap.

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
training composition but does not call it NEMO bit-for-bit. The user accepted
this debt on 2026-10-13; the unchanged `1e-15` certification bar still records
it as debt.

The state target is the conservative restriction in `swg_coarse_graining.py`:
sum fine T-cell volumes into each coarse T cell and sum `e2*h*u` or `e1*h*v`
over the fine faces tiling each coarse face, then diagnose coarse SSH and face
velocity using the coarse live QCO thickness. There is no component-mean
restoration. The real day-1095 continuity residual is `4.520e-18 m s-1`
interior and `2.887e-18 m s-1` in the outer four cells. The round-1
independent-state Gaussian remains a labelled diagnostic and is forbidden as a
training target. New daily-mean acquisitions accumulate instantaneous `h*u`
and `h*v` inside the compiled scan and restrict those mean transports; archived
`h(mean eta)*mean(u)` products are explicitly refused. A temporal continuity
plant that offsets restricted SSH by 1 m fails.

There are two named objectives and **no default**. `symmetric` is the Round-4
departure: initialize and target with the exact box restriction above, then
apply a sigma-one-coarse-cell Gaussian to both model and target vorticity.
`template` follows Maddison's data/model asymmetry: initialize and target from
one fine-grid filtered restriction, while the model vorticity in the loss is
unfiltered. Maddison defines the Gaussian “width (specifically a standard
deviation)” as one coarse spacing, filters instantaneous fine vorticity, and
then samples the coarse grid ([Maddison 2026, sections 2.3--2.4](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004883)).

The `template` state operator works on finite-volume extensives before exact
restriction. Fine T-cell volume is even-reflected about every wall. U transport
is even across the south/north walls and odd about its zero west/east wall
faces; V transport is odd about its zero south/north wall faces and even across
the west/east walls. To see why, extend the closed-grid continuity stencil
`D(Q)=Qe-Qw+Qn-Qs`: an even scalar extension requires the normal face flux to
change sign, whereas a tangential flux remains even. The same separable
convolution therefore satisfies `D(G*Q)=G*D(Q)`, fixes normal wall transport to
exact zero, and preserves total volume without a posteriori mean restoration.
The Gaussian has standard deviation 50 km = 8 fine cells and transfer
`exp(-2 pi^2 sigma^2/lambda^2)` for an interior plane wave. The filtered
extensives are then restricted by the existing exact volume/face sums.

The reference is not yet adequate. At 6.25 km the Munk width has
1.27 cells; restricted daily-mean truth gives consecutive annual mean-velocity
differences of 0.04765 and 0.04203 m s-1 for years 1-to-2 and 2-to-3. A 3.125 km
sensitivity is therefore mandatory. Its first eight days are compared from the
common rest state using normalized RMS (bar 10%) and pattern correlation (bar
0.99) for coarse SSH, relative vorticity, and rotational velocity, with first 1% and 10%
crossings reported. Year-1 KE, EKE, jet position, and DST-streamfunction KE
spectrum must each agree within the larger of 10% and measured adjacent-year
variability of the 6.25 km run. The tool exits nonzero on inadequacy.

## Initialization and hourly observations

Ten starts are fixed at fine days 731, 771, ..., 1091. Each supplies both
four-day coarse states in one compiled fine-model pass: exact box restriction
and filtered restriction. Schema v2 stores T/U/V at t0 plus 24 right endpoints
per day for each operator; schema-v1 readers remain box-only. Enabling either
sampler must leave the prognostic path and daily restart words unchanged. A
restarted segment must reproduce the uninterrupted next fine daily state
bit-for-bit under the same fp64/native policy. Maddison's loss has `M=24w`
hourly outputs; this lane likewise scores 24, 48, 96, or 192 right endpoints
for 1, 2, 4, or 8 days.

Before training, every start receives a 48-hour no-network dg50 forecast from
each candidate state and is compared with targets made by that same operator.
Admission applies separately per start: vorticity error at hour 1 must be at
most twice the hour-24 error divided by 24, and maximum fast divergent KE in
hours 1--6 must be at most twice its hours-25--48 mean. Truth fast energy,
truth vorticity RMS, and SSH tendency at hours 1, 6, 24, and 48 are reported.
If at least 8/10 filtered starts pass, experiment one selects `template`;
otherwise it selects `symmetric` and records the measured departure. The old
result (box-conservative 10/10, raw independent filter 0/10) remains diagnostic
only and does not prejudge the filtered-restriction decision.

## Closure seam and pytree contract

The closure is an Equinox module producing one scalar potential on native
F points. The card applies the matched rotated gradient to form native U/V
accelerations at **every RK3 stage**. The same F-point potential is constant on
the closed boundary, so wall-normal force is zero; the discrete divergence of
the volume-force transport is roundoff zero. The seam is a pure pytree term
inside the certified step. If the closure argument is absent, no addition or
reassociation occurs, preserving the certified card byte-for-byte.

The orientation is explicit: positive U is east and positive V is north. A
positive northward potential difference accelerates U eastward, while a
positive eastward potential difference accelerates V southward. Thus the force
is the clockwise rotated gradient `(dPhi/dy, -dPhi/dx)`. Boundary potential is
zero at all vertices, including corners.

Inputs are coarse SSH, native U/V velocity averaged to T, wet mask, normalized
non-trainable C-grid metrics, an explicit RNG key, and a fixed-shape noise
pytree. Uniform metrics act only in the potential-to-face transform and are
not duplicated as indistinguishable CNN channels. The
phase-3 Equinox model has five width-3 convolutions, 16 hidden channels, ELU,
an 11-cell (550 km) receptive field, and 7,696 parameters. Its output is the
native F-point potential; U/V acceleration is derived, never independently
predicted. The final convolution has no bias because a constant-potential bias
would be annihilated by the curl and become inert. Noise enters the SSH channel
through the same state weights, so deterministic zero noise leaves no
noise-only trainable leaf. This is intentionally smaller than Maddison's seven
width-5, 32-filter residual convolutions with 154,857 weights (paper lines
436--442). Architecture and normalization scales are mandatory manifest data,
not training defaults. Initialization uses seed 20261013 and is small random,
never exactly zero. On ten real t0 states its maximum closure acceleration is
`1.519e-9 m s-2`, or `0.7792%` of `tau0/(rho0 H)`, below the pre-registered 1%
bar while leaving every trainable leaf live.

The closure module, noise array, noise scale, and RNG key are explicit traced
arguments. Per the round-3 ruling, the analytically steady wind is closed over
by the segment builder (it is neither time-varying nor trainable). Separate
functions are built once for each admitted window. Replacing parameters,
noise, or keys at fixed shapes does not retrace.

Before training, tests must fail on planted violations of
F-to-U/V staggering, zero wall-normal force, divergence, integrated momentum,
absence-is-exactly-zero, and JVP/finite-difference agreement. Static masks and
metrics are not trainable. Every trainable leaf must have a finite nonzero loss
gradient under the repository's no-inert-parameter gate. The gate runs through
one real 24-hour rollout, real loss, and AdamW update from a content-pinned
conservative start; one-step and 24-hour JVPs agree
with centred finite differences, and closure-free 4/8-day segment outputs are
bit-identical at their 96 common hourly endpoints.

## Loss, rollout, and curriculum

The objective is an explicit required argument. For `template`, fine volume and
transport are filtered before restriction, that stored state is both t0 and
truth, and the model vorticity remains unfiltered. For `symmetric`, stored box
restriction is t0 and truth and the same coarse Gaussian acts on both diagnosed
vorticities. In either case loss is exactly Maddison's Eq. 7: squared vorticity
error weighted by `dx^2` at interior F points, multiplied by
`1e4/(beta^2 L^2)/(4 L^2)`, and averaged inside the loss over all `M=24w`
hours and all `N_windows` batch members. Consequently the loss is exactly zero
at a stored window start, while the template objective still sees model-created
grid noise. Rotational-velocity terms are explicit options and off in
experiment one. Raw SSH is monitored, not fitted.

One content-pinned dataset contract owns source-root and manifest hashes,
operator name, split bounds, and batch construction. Each T0 is the stored
T/U/V state at the chosen absolute hour and its targets are the following
right endpoints from the same operator. Splits never share an hour and their
endpoints are separated by at least 720 hours. The planned 12-year record is
`[day731, day5111)`: training is `[731,4235]` (3,504 days, the template's
first 80%), a 30-day gap precedes validation `[4265,5111]`, and another
30-day gap precedes the independent test year `[5141,5506]`. Normalizers are
computed only by the contract's training-split method and record the contract
and source hashes. Any held-out normalizer request, split-crossing window,
unregistered manifest, duplicate hour, or coverage gap inside a split raises.

At each epoch and window length the contract redraws one seeded starting
offset, forms non-overlapping 1/2/4/8-day windows, shuffles them, and yields
batches of exactly ten. The 24 complete schema-v1 eight-day segments available
at the Round-5 check are accepted only as pinned box diagnostics and prove the
loader handles 192-hour inputs; they cannot silently supply the filtered
objective or a missing t0. The future schema-v2 record is the training source.

The first curriculum is exactly 1, 2, 4, and 8 days. A selected checkpoint
initializes the next window. No 16, 32, or 60-day deterministic stage is
authorized until the real network, batch, loss, and optimizer have been
profiled. For scale, 60 days is 12,960 coarse steps and projects to about
22.1 s and 4.49 GiB per sample under uniform checkpointing, beyond the measured
balanced-vorticity predictability time. The pre-registered balanced fit gives
10.282 days (`1e-11 s-1`, R2 0.957) and 13.621 days (`1e-10 s-1`, R2 0.885);
the amplitude dependence is reported rather than averaged away. Uniform
production-step checkpointing
is the default; alternatives require a measured memory/time receipt.

Optimizer construction uses `legoesm.ml.training.create_optimizer`. Experiment
one passes the template settings explicitly: constant learning rate `1e-3`,
AdamW weight decay `0.004`, beta1 `0.9`, beta2 `0.999`, epsilon `1e-7`, no
separate clipping transform, batch 10, and five epochs per window. The network
is carried across 1/2/4/8-day stages and optimizer state is reinitialized at
each stage. The shared factory's historical defaults remain bit-for-bit the
prior warmup/cosine, clipped, epsilon-`1e-8` result for every existing caller.
Optax 0.2.8's AdamW source and the factory were read; a hand-computed first
update guards the explicit experiment settings.

Learning rate, schedule, epsilon, clipping, batch size, start sampler, split
dates, architecture, and parameter count are manifest fields, not defaults. The
truth-generation day scan and training segment loop are distinct named
compositions. With zero closure, 4-day and 8-day runs from the same coarse state
must match bit-for-bit in native T/U/V at every common hourly endpoint. A
failure blocks training rather than widening a tolerance.

The phase-3 CPU profile used the real 7,696-parameter network, one sample, and
uniform production-step rematerialization. These are execution-only profiles;
the temporary unit-valued field scales are not admitted training
normalizations and no optimizer update was taken.

| Window | Steps / observations | Forward execution / peak RSS | Reverse execution / peak RSS |
|---:|---:|---:|---:|
| 1 day | 216 / 24 | 2.15 s / 1.10 GiB | 10.58 s / 1.86 GiB |
| 2 days | 432 / 48 | 4.71 s / 1.18 GiB | 14.12 s / 2.05 GiB |
| 4 days | 864 / 96 | 8.50 s / 1.33 GiB | 30.24 s / 2.39 GiB |
| 8 days | 1,728 / 192 | 17.38 s / 1.62 GiB | 62.67 s / 3.10 GiB |

## Controls and held-out gates

Both no-closure controls (`nu=10` Maddison replication and `nu=80`
grid-Reynolds control) use the same evaluation harness. The `nu=80` control
requires a finite, positive-thickness, sub-unit-Courant 30-day preflight before
its 20-year CPU run. Reports are yearly, map scores use wet cells only, and
health includes top-octave DST-streamfunction KE, direct-vorticity enstrophy,
and a local second-difference grid-noise index that a phase reversal cannot
evade. Years 2–3 are currently the least-confounded exploratory
window, not an equilibrated claim.

Each trained window has at least three network seeds and a 20-year free run.
Every seed must remain finite with positive thickness and `Cu,Cv < 1`. On the
held-out years, the median seed must beat the **better** of the two controls in
live-face-transport absolute error, KE-spectrum relative L1 error, and wet-cell EKE
map normalized RMSE. Every seed must retain basin EKE within 20% of filtered
truth and must not exceed the worse control's top-octave fractions or
grid-noise index. Uncertainty ranges and raw seed scores are reported; a
median alone cannot hide an unstable seed.

The corrected native-face scorecard freezes control manifests and code
thresholds. On years 2--3, `nu=10` has live-face-transport RMSE
`1.439733e6 m3 s-1`, DST-spectrum L1 `0.223004`, wet EKE-map NRMSE `0.393673`,
and EKE ratio `0.732341`; `nu=80` has `2.007138e6`, `0.386407`, `0.773158`, and
`0.396429`. Native-face `nu=10` top-octave KE, direct-vorticity enstrophy, and
second-difference bars are `0.0538625/0.435918/0.931792`; `nu=80` gives
`0.0275464/0.294723/0.706924`. The better skill and worse noise control are
therefore both `nu=10` for these registered fields. Finite fields, positive
thickness, and sub-unit Courant are unconditional per-seed gates; a perfect
statistical score cannot override failed physics.

Two adversarial controls are mandatory. Replacing every time-varying anomaly
by its time mean must fail the EKE-retention gate (collapsed variance).
Adding a native two-cell checkerboard in either phase above the worse control must
fail spectral health. A gate release that lets either plant pass is itself a
failed gate. Mean transport merely lying between a baseline and truth is not
an acceptance criterion.

## Stochastic branch

Stochastic readiness is **interface only in phase 4**. The rollout API carries an explicit RNG key and member-indexed noise
pytree. The loss API accepts `S` trajectories and dispatches between point loss
and energy score without changing solver or closure signatures. The dormant
energy-score implementation follows Brolly Eqs. 9--10: Euclidean score at each
lead, pair term divided by `2 S (S-1)`, then uniform lead averaging. It requires
`S >= 2`; `{truth, truth+1}` scores exactly zero. No stochastic rollout or
stochastic-learning claim is made this round.
There is no inactive trainable “noise parameter” in deterministic mode. The
noise scale is an explicit non-parameter input; noise uses the same weights as
the state path. Any future choice to make its scale trainable must first pass
the no-inert-parameter gate. Collapsed ensemble variance is a planted failure.

## Review requirements and enforcement

| # | Required change | Design answer | Enforcing test or measurement |
|---|---|---|---|
| 1 | Conservative, balanced initialization | Explicit box/symmetric and filtered/template states; measurement selects one | Per-start no-jump/fast-energy gates; filtered requires at least 8/10 |
| 2 | Hourly targets | Both coarse states at t0 and 24 right endpoints/day from one fine scan | Ten dual four-day caches; sampler-off/both-mode and restart-next-day word gates |
| 3 | Adequate reference | Early common path plus uncertainty-aware 3.125 km sensitivity | Days 1--8 NRMS/correlation and year-1 statistics against interannual variability |
| 4 | Balanced predictability | Smooth zero-divergence vorticity perturbation, two amplitudes, JVP | 120-day twins and registered balanced fit rule |
| 5 | Cap curriculum | Initial windows 1/2/4/8 days only | Network-inclusive profile required before 16/32/60 days |
| 6 | Non-divergent forcing | F-potential rotated gradient at every RK3 stage | Divergence, walls, momentum, zero seam, JVP/FD plants |
| 7 | Viscosity controls | `nu=10` and `nu=80` are separate named controls | 30-day preflight, then identical 20-year health/climate harness |
| 8 | Structural stochastic hook | Interface only: explicit RNG/noise and swappable Brolly score, `S>=2` | Counterexample, RNG independence, inert-noise and collapse plants |
| 9 | Quantitative multi-seed gates | Three seeds; beat best control; variability and noise bounds | Held-out harness plus collapsed-variance/grid-noise failures |
| 10 | Correct physics and composition | Full-gravity explicit free surface; named truth/training loops | Zero-closure 4/8-day bitwise A/B and manifest composition fields |

## Round-5 implementation disposition

| Requirement | Design answer | Test or measurement |
|---|---|---|
| F-potential seam | Matched live-thickness transport curl at every RK3 stage; absent branch makes no addition | Random-potential divergence/wall/integral tests; unchanged dg50 ladder |
| Explicit stochastic structure | RNG, member noise, and scale are traced; shared weights avoid an inactive noise branch | RNG/no-retrace tests and no-inert gradient gate |
| Objective choice | `template` is fine-filtered target versus unfiltered model; `symmetric` is explicit departure; no default | Zero-at-t0, checkerboard response, analytic transfer, and 8/10 admission rule |
| Initial curriculum | Builders admit only 1/2/4/8 days | Refusal test for 16 days and measured per-window profiles |
| Segment composition | Flat hourly scan with uniform per-step rematerialization; steady wind closed over | 4/8-day zero-closure common-endpoint bit identity |
| Dataset | Pinned schema-v2 roots/hashes; split-owned normalizers; seeded non-overlap batches of ten | Eight-day real loader and validation/test/cross-boundary leakage plants |
| Optimizer | Exact template AdamW through repository `create_optimizer`; historical defaults unchanged | Hand first update, bitwise legacy update, real pinned rollout/loss/update |
| Multi-seed gate | Median beats better control on three skill measures; every seed meets variability/noise bounds | Three-seed minimum plus failed collapsed and grid-noise plants |
| Coarse controls | `nu=10` and `nu=80` scored from native arrays on years 2--3 | Frozen hashes, live transport, uncertainty, and four failing plants |
| Model identity | Exact and numerical composition scores are kept separate | Three 30-day artifacts with first-cell ULP localization |
| Training release | User accepted composition debt; filtered-start decision, adequate reference, and pinned data remain | Explicit BUILD-NOT-TRAIN hold; only the existing single-step seam test |
