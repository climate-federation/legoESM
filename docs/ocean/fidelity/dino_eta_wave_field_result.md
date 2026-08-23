# DINO sea-surface wave-field twin — legoESM vs NEMO, sampled every model step

**The question.** Do the first-days sea-surface wave patterns match between the
two models? Both start from the same NEMO day-180 restart and single-step
comparisons match closely, but the propagating free-surface field had never
been compared at its own timescale: the saved twin output was DAILY (32 model
steps) while a barotropic gravity wave crosses the DINO basin in about four
hours.

**The short answer.** The wave patterns match in SHAPE and in the frequency
domain — spectral peaks at identical frequencies, peak amplitudes within 0.5 %,
no band carrying excess energy — but they are OFFSET IN TIME:

> **legoESM's barotropic impulse response lags NEMO's by 0.47 of a baroclinic
> step, about 21 minutes.** One scalar explains 71 % of the entire
> model-to-model residual in the impulse run. The free run, used as a control,
> shows no lag (0.02 steps, explaining 0 %).

That is the finding. Getting to it took a second experiment, because the free
run has no waves in it at all, and a better estimator, because a half-step lag
is invisible to any test that can only score whole steps.

---

## Two retractions, up front

**1. The premise of the run was wrong, and so was my first explanation of why.**
The daily record was not aliasing the basin's barotropic waves — there are no
such waves in the free run to alias. NEMO's own free surface at day 180 moves
1.8e-4 m per step against an 0.83 m field, and 99.99 % of its temporal variance
sits at periods longer than a day. The basin radiated its free gravity-wave
energy away during 180 days of spin-up under smooth seasonal forcing. I then
attributed that to the split-explicit time-averaging filtering the waves out;
**that is also retracted** — the 90-minute averaging window passes 94 % of an
8-hour signal, so it is not what removed them.

**2. I claimed the two models propagate at the same speed to 0.13 %. That is
withdrawn.** The spreading-rate measure saturates: a uniform field has a mean
radius of 4542 km and NEMO's response passes half of that within two samples of
160, after which both models simply say "the basin is full" and the ratio is
forced to 1. The median I quoted was that forced number.

**3. I then wrote that the timing question was OPEN and that settling it would
need the free surface sampled inside the barotropic substep loop. Both halves
are withdrawn.** The question was not open — my discriminator could only score
whole-step offsets, so a half-step lag made every candidate equally bad and
produced the 44 % coin toss I reported as inconclusive. **"Inconclusive" was
the signature of the answer.** And it needed no finer output: fitting the lag
as a continuous quantity settles it from the same 45-minute samples.

**4. The time-level identity I first wrote was wrong, and its residual was
nearly read as physics.** The check on which dump corresponds to which time
level was written as

    sshb[n] = sshn[n-1] + a*( sshn[n-2] - 2 sshn[n-1] + sshn[n] )      WRONG

which leaves a residual that looks like a small physical correction — 1.3 % of
a step's change on the quiet run — and was about to be explained as a
forcing-exclusion term. The level entering the filter is the already-filtered
one:

    sshb[n] = sshn[n-1] + a*( sshb[n-1] - 2 sshn[n-1] + sshn[n] )      RIGHT

In that form the residual is **exactly 0.0**, to the last bit, on both NEMO
runs. There is no physical residual. The wrong form was caught only because it
failed on the excited run, where the same second-order error is 16 % rather
than 1.3 %.

---

## What was run

Five days (160 steps of 2700 s) from `DINO_00005760_restart.nc`
(md5 `ad5ba426…`), free surface saved EVERY step on both sides — 45-minute
sampling, finer than the hourly asked for, because an hourly cadence is not an
integer number of NEMO steps.

| arm | NEMO | legoESM |
|---|---|---|
| FREE | certified binary, `RUN_90D_TWIN` namelist with only `nn_itend` and `nn_stock` changed, 16 ranks, `STOP 0` | twin harness, fp64, 74 s on one GPU |
| IMPULSE | same namelist, only the restart differs | same, only the restart differs |

The impulse arm adds an identical 0.05 m Gaussian sea-surface bump (300 km
e-folding radius, centred 30°S / 25°E, applied to both `sshn` and `sshb` so the
leapfrog starts consistent) to the shared restart. It is coordinate-safe: the
DINO restart carries only `sshb`/`sshn` for the free surface and the vertical
ladder comes from the static mesh, so both models rebuild the stretched
coordinate from the ssh they read. The generator refuses to run if it finds an
ssh-dependent thickness field in the restart.

legoESM's day-0 gate reported `max|dT| = max|d_eta| = max|du| = max|dv| =
0.000e+00` against the restart in every arm, so the two models start from
bit-identical data.

Instrument: `scripts/validate/ocean_fidelity/dino_1226/eta_wave_twin.py`.

---

## Time centering

NEMO swaps its time-level indices at `MY_SRC/stpmlf.F90:621-624` BEFORE calling
`rst_write(kstp, Nbb, Nnn)` at `:634`, and calls it without a `Kaa` argument,
so the MLF branch `restart.F90:197` writes `sshn` from the POST-swap `Kmm` —
the pre-swap AFTER level, i.e. **the sea surface height after `kt` steps**, and
unfiltered. The same ordering is in the untouched upstream
`src/OCE/stpmlf.F90:402`, so it is core NEMO rather than a local edit. On the
legoESM side the Asselin result is parked separately and the saved `eta` is
likewise the unfiltered after-level, so the comparison is like with like.

Two independent confirmations: the exact Asselin identity above (residual
0.0), and a four-way scoring of the registered pairing against its
alternatives. The second one is **UNRESOLVED and reports itself as such** — the
four candidates are separated by 3.7e-6 m while NEMO's own two time levels sit
1.8e-5 m apart, so the empirical test cannot see the difference it is asked
about. The source citation and the exact identity are what carry the claim.

---

## Result 1 — the free run: no waves, and none expected

Share of temporal variance over wet cells, area-weighted, 45-minute sampling:

| period band | NEMO | legoESM | their difference |
|---|---|---|---|
| < 6 h | 2.8e-06 | 3.3e-06 | 0.0053 |
| 6–24 h | 7.0e-05 | 8.3e-05 | 0.0277 |
| > 24 h | 0.99993 | 0.99992 | 0.9670 |

Both models put 99.99 % of their free-surface variance at periods longer than a
day, and at all four probe points the spectral peak sits in the lowest resolved
bin — the record length, not a wave. Only two spectral bands clear the
live-band threshold at all.

A second, independent measurement agrees. Along the ACC channel row (54.9°S,
51 adjacent wet pairs, 63.9 km cells) the lag that maximises the
cross-correlation between neighbouring cells is **exactly zero steps in both
models and in their difference**, 51 pairs out of 51 within one step. That is
the arithmetic one expects: a barotropic wave at sqrt(gH) = 210 m/s crosses one
cell in 304 s, 0.11 of a time step, and the whole 3323 km basin in 4.4 h.
**Cell-to-cell phase lag is unresolvable at baroclinic-step output for this
basin, in either direction** — no wave-speed claim, agreement or disagreement,
can be made from it. Propagation is resolvable spatially (where the front has
reached at each sample), not temporally.

**What the free run does say.** The two free surfaces track closely: the day-5
difference is 2.6 % of NEMO's own five-day change at the worst cell and 2.3 %
in rms, and it saturates near 8e-4 m around day 3 rather than growing. Against
the pre-registered bound of 160 × the one-step floor it is 0.14×.

Two things worth naming anyway:

- **The one-step agreement is not roundoff.** After a single step the two free
  surfaces differ by 3.4e-5 m where NEMO's own one-step change is 2.2e-4 m —
  the free-surface increment disagrees by 19 % at the worst cell and 36 % in
  rms. "Single-step comparisons match to tiny levels" is true of the tendency
  terms; it is not true of the assembled free-surface increment. That the
  accumulated difference nonetheless stays at 2 % of the signal means those
  per-step disagreements largely cancel.
- **legoESM's free surface carries its OWN two-step mode, on the walls.**
  Measuring the alternating component directly over the first eight steps:
  NEMO 3.9e-7 m, legoESM 7.0e-6 m — **18× the oracle** — against leakage floors
  of 9.9e-8 and 1.3e-6 m, so both are above their floors and the above-floor
  ratio is 19.5. It decays to 1.4× over the second half of the run.

  Quote it as **18–20×**: 17.95 on the raw amplitudes, 19.50 after subtracting
  each side's own leakage floor. The floor is a similar fraction of both sides
  (25 % and 19 %), so it largely cancels and the ratio is insensitive to how it
  is handled.

  Two qualifiers that matter. Its signed spatial correlation with NEMO's is
  **−0.09 to +0.08, i.e. zero**: unlike the impulse lane, legoESM is not
  ringing the oracle's mode harder, it is ringing a different one. And it is
  **not basin-wide** — the wall enrichment of its first sample is 9.4, so this
  is a BOUNDARY transient, and calling it "the leapfrog computational mode
  under-damped by 18×" would overstate it. In absolute terms it is small
  (7e-6 m on an 0.83 m field) and it decays. Unlike the impulse lane's
  alternation, this one is NOT a re-description of the lag: the free run has
  no lag.
- **That excess sits on the walls.** Per-cell ratio of mean-squared step-to-step change:
  median 1.06, 90th percentile 2.59, maximum 57.6. Splitting the excess by
  region, area-weighted, gives an **enrichment of 7.5× on land-adjacent cells**
  (28 % of the excess in 3.8 % of the area), 0.77× in the interior and 0.18× at
  the equator; the wall enrichment of the difference field itself climbs from
  1.6 at the first sample to 8.0 by day 2. The difference field is also far
  less red than either signal: 3.3 % of its variance is at periods under a day,
  against 0.007 % for the signals.

---

## Result 2 — the impulse run: this is where the wave answer is

Adding the bump does what it was meant to. NEMO's response carries 16.5 % of
its variance in the 6–24 h band, against 0.007 % in the free run.

**Spectral peaks match exactly, and amplitudes to under one per cent:**

| probe | peak period, NEMO | peak period, legoESM | peak amplitude NEMO / legoESM | worst band ratio |
|---|---|---|---|---|
| channel (54.9°S) | 30 h | 30 h | 1.030e-4 / 1.035e-4 m | 1.04 at 10 h |
| equator | 20 h | 20 h | 4.796e-5 / 4.847e-5 m | 1.17 at 6.3 h |
| west wall | 30 h | 30 h | 1.697e-4 / 1.703e-4 m | 1.06 at 60 h |
| mid-basin | 30 h | 30 h | 1.302e-4 / 1.306e-4 m | 1.11 at 6 h |

Every band ratio is inside the pre-registered factor of 2, evaluated in code
rather than by eye, over 11–21 live bands per probe. **No band carries excess
legoESM energy.**

### The response is half a step late

A lagged trajectory is, to first order, a linear interpolation between two of
the reference's samples, so fitting

    eta_lego[k] - eta_nemo[k]  =  alpha * ( eta_nemo[k-1] - eta_nemo[k] )

by area-weighted least squares over wet cells and samples returns the lag
directly, as a continuous number of steps. Measured:

| lane | lag alpha | in minutes | share of residual variance explained |
|---|---|---|---|
| **impulse, all 160 samples** | **+0.470 steps** | **+21.2 min** | **0.714** |
| impulse, first 8 samples | +0.458 steps | +20.6 min | 0.679 |
| free (control), all samples | −0.132 steps | −5.9 min | 0.004 |
| free (control), first 8 | +0.018 steps | +0.8 min | 0.000 |

Positive means legoESM is behind. The competing explanation — that legoESM's
field is simply scaled rather than delayed — was fitted alongside and explains
0.1 % of the same residual against the lag's 71 %. The free run is the control
that matters: the same estimator, the same two models, no wave in the field,
returns no lag and explains nothing, so the fit is not manufacturing a number
out of any two imperfectly matching trajectories.

**Four things reported below are re-descriptions of this one lag, not
independent findings**: the step-to-step alternation of the difference (odd
samples 5.1× the even ones), the sample-1 spreading ratio of 0.713 against
0.995 at sample 2, the 2-step ringing in the impulse lane, and the 44 %
bootstrap. A half-step offset sampled on alternate steps produces all four.

**PLAUSIBLE mechanism, not confirmed.** Half a step is exactly what a
disagreement about where the barotropic time-average is CENTRED would produce
— an average over `[t, t+dt]` sits half a step later than one over
`[t-dt/2, t+dt/2]`. This campaign already has history on that exact window: the
two barotropic averaging kernels (the velocity boxcar and the
transport-weighted tail-sum) are recorded in `ocean/state.py` as sampling the
substep profile at different phases, with centroids 7.3 substeps apart out of
23 — about 0.32 of a step. Right family, right order, not equal to the 0.47
measured here. The discriminating test is cheap and has not been run: compare
the two kernels' centroids at the twin's actual substep count and see whether
the difference matches. Until then this is a labelled guess; the lag itself is
measured.

**What is still not measurable here.** Cell-to-cell phase lag (0.11 of a step
per cell) and the spreading-rate comparison, which saturates after two samples
— its usable ratios are 0.713 and 0.995, and the whole-run median of 1.002
means nothing.

**The response difference decays.** As a fraction of the response NEMO itself
produced:

| t | max abs difference | rms difference | rms as a fraction of the response |
|---|---|---|---|
| 0.75 h | 5.79e-03 m | 1.01e-03 m | 1.22 |
| 2.25 h | 1.67e-03 m | 5.29e-04 m | 0.84 |
| 3.75 h | 1.58e-03 m | 3.60e-04 m | 0.85 |
| 8.25 h | 3.84e-04 m | 1.06e-04 m | 0.31 |
| 12 h | 1.70e-04 m | 6.29e-05 m | 0.19 |
| 24 h | 9.94e-05 m | 3.48e-05 m | 0.15 |
| 48 h | 1.77e-05 m | 8.46e-06 m | 0.032 |
| 120 h | 2.37e-05 m | 8.44e-06 m | 0.038 |

Unlike the free run, the excess in this lane is NOT on the walls: the
enrichment of the step-to-step excess is 0.13 on the walls, 0.11 at the
equator and 1.06 in the interior. legoESM's wall problem is a property of how
it carries the balanced background, not of how it propagates a wave.

**The early disagreement is the leapfrog computational mode, and here the two
models agree about it.** The difference alternates sign step to step: odd
samples average 5.1× the even ones over the first eight steps, falling to 1.37
afterwards. Both models ring at the two-step period after an impulsive
displacement, which is what leapfrog does. Measuring the alternating component
directly: over the first eight steps NEMO's is 1.283e-4 m and legoESM's
1.369e-4 m — **legoESM 1.07× the oracle** — against leakage floors of 5.8e-5
and 6.3e-5 m, so both sit well above the floor and the above-floor ratio is
1.05. Their signed spatial correlation is **+0.83 to +0.97 on odd samples**:
the two models are ringing the SAME mode, not each its own. And it is
basin-wide here (interior enrichment 1.08, wall 0.12). When a genuine wave
dominates the signal, legoESM's computational mode is not under-damped.

Note also that the bump is largely gone after ONE baroclinic step: NEMO's
response drops from 2.3e-3 m rms at the start to 8.3e-4 m after one step, and
at the bump's centre the response has already reversed sign. With 30
barotropic substeps per step the wave travels ~9 cells inside a single step, so
the radiation is a sub-step process here. What the sampled record shows is the
adjustment after that, settling to a ~2.2e-4 m basin-scale pattern that the two
models agree on to 4 %.

---

## Verdict against the pre-registration

Registered before the comparison ran: CONFIRM if (i) the difference stays
within 1.0 × (160 × the one-step floor) and (ii) no resolved spectral band
shows a ratio beyond 2×.

- **FREE lane: (i) PASSES at 0.14×. (ii) PASSES**, but only two bands are live,
  so it is close to vacuous — there is nothing to compare.
- **IMPULSE lane: (i) NOT APPLICABLE** — the first sample is the largest in the
  series, so 160 × the floor cannot be exceeded by construction. The instrument
  reports this as null with a reason rather than as a spectacular pass.
  **(ii) PASSES with margin** — worst ratio 1.17 of an allowed 2.0, across
  11–21 live bands per probe, and that is the criterion the verdict rests on.

**CONFIRMED: the sea-surface wave patterns match in the frequency domain.** The
claim rests on criterion (ii) in the impulse lane — the only lane with a wave
in it — and on the response difference falling to 3.8 % of the response by day
5. It does NOT rest on the free lane, whose registered pass is an artefact of
there being no signal, nor on criterion (i) in either lane.

**CONFIRMED, and the more consequential half: legoESM's barotropic response is
0.47 of a step late** — 21 minutes, explaining 71 % of the impulse-lane
residual, with a free-run control at zero. The shapes agree; the clocks do
not. What remains unmeasurable here is cell-to-cell phase lag and the
spreading-rate comparison, both below resolution or saturated.

**The feedback-tier finding is the free lane's**, and it is separate from the
wave question: in a free run legoESM's free surface carries its OWN two-step
mode — 18× the oracle's amplitude over the first eight steps, uncorrelated
with the oracle's, concentrated on the boundary rows (wall enrichment 9.4),
decaying to 1.4× — and the step-to-step excess is enriched 7.5× on
land-adjacent cells, peaking at ~58× on one cell of the eastern wall. When the
impulse gives it a real wave to carry, the same measure is 1.07× and the two
models' modes are strongly correlated. This is the locus the
campaign's wall-row velocity work has been circling.

---

## Controls run before any number above was believed

- **Time level.** Cited in source; verified by NEMO's exact Asselin identity
  (residual 0.0 both arms); a synthetic one-dump offset is planted in a fake
  NEMO run and the extractor is required to refuse it. The four-way empirical
  discriminator reports itself UNRESOLVED rather than pretending to settle it.
- **Mask.** From the model's own `mesh_mask` surface `tmask` (9920 wet, 428 dry),
  never from "where the field is zero" — that would delete genuinely zero ocean
  values and the equator. Every dry cell is then poisoned with 1e6 and the run
  aborts unless every masked statistic is bit-identical on the poisoned copy
  AND the unmasked one moves. Both halves passed.
- **Periodicity.** DINO is zonally periodic and its east–west walls are land
  columns, so the zonal neighbour test wraps; the earlier version mislabelled
  the re-entrant channel's 70-cell seam as wall, in the very statistic used to
  name a locus.
- **Locus width.** The wall band (1 cell) and equator band (2°) are choices, so
  the split is reported at four equator widths (2/5/10/26°, the last being the
  barotropic equatorial radius) and two wall widths.
- **Area weighting.** All global statistics weighted by cell area; the basin
  spans 70°S–70°N.
- **Spectra.** Window-consistent demean; the coherent-gain correction halved at
  the zero and Nyquist bins (Nyquist is where a computational mode would sit);
  the live-band cut symmetric over both models so a band where legoESM rings
  and the oracle is silent cannot be excluded.
- **`rn_Dt`.** Read back from every NEMO dump and required to be 2700 s.
- **Precision.** The comparison refuses a float32 artifact; the free surface is
  stored at fp64 on both sides.
- **NaN.** Fatal in the statistics path; the only nan-reductions are in figure
  colour scaling, on data already proven finite.
- **Mesh identity.** The mesh supplying the equator band and the four probe
  points is checked against the NEMO artifact's own coordinates.
- **Estimator floors.** Every 2-step amplitude is quoted next to the leakage
  floor of its own estimator — the operator is a curvature high-pass, not a
  notch, so about 29 % of a 6-hour signal passes it and a ratio taken without
  the floor can be mostly leakage.
- **Lag estimator.** Validated on a planted ladder: lags of 0, 0.25, 0.5 and
  0.75 steps are recovered exactly, the sign flips when the two models' roles
  are swapped, and a uniformly scaled field with no lag is correctly reported
  as an amplitude error rather than a lag.
- **Instrument tests.** 66 direct unit tests, written as known-answer and
  synthetic-violation controls: the spectrum recovers a 0.37 m / 8 h sinusoid to
  5 % and a Nyquist oscillation at its true amplitude; a planted wall-band
  2-step mode is detected and attributed to `wall` with >99 % of the excess; a
  planted travelling wave returns its exact lag; the two-step-mode metric
  annihilates a smooth ramp to 1e-12 and returns a planted sign-flip's own
  amplitude; removing the zonal wrap turns
  a seam cell into a wall and a test goes red; a float32 artifact, a mismatched
  mesh, a one-sided impulse lane and a one-dump time offset are all refused;
  `compare` itself is exercised end to end on a synthetic mesh and artifact
  pair; a NEMO run whose files carry a wrong internal step counter is refused,
  as is one with a missing tile.

- **Registration.** The Asselin identity is a recurrence in the array index and
  is therefore invariant to a uniform shift of the whole window — it pins
  `sshb` against `sshn` and the direction of time, but NOT which file is step
  N. That is pinned separately from each dump's own step counter. Neither
  check alone closes the question; together they do.

---

## Figures

`/tmp/dino_eta_waves/figs/free/` and `/tmp/dino_eta_waves/figs/impulse/`, each
with `fig_diff_maps.png` (difference maps at the eight pre-registered times),
`fig_hovmoller.png` (time–longitude at the channel latitude and time–latitude
at mid-basin longitude, both models and their difference), `fig_growth_locus.png`
(growth curve against the linear bound, plus locus enrichment on a log axis)
and `fig_spectra.png`. Full numbers in each directory's `eta_wave_twin.json`.

## What this does NOT establish

- Nothing about barotropic wave PHASE SPEED as a propagation rate. The 0.47-step
  result is a whole-response timing offset, not a dispersion measurement.
  Cell-to-cell lag remains unresolvable (0.11 of a step per cell) and the
  spreading-rate comparison saturates after two samples.
- Nothing about the CAUSE of the lag. The averaging-centroid mechanism above is
  labelled PLAUSIBLE and its discriminating test has not been run.
- Nothing about the 78–435× kick-amplification result that motivated the run.
  That measures the growth of an injected perturbation over thirty days; this
  measures the difference between two models over five.
- Nothing below ~2e-5 m in the free lane, which is where NEMO's own two time
  levels sit apart.
