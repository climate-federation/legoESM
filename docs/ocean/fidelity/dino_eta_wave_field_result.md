# DINO sea-surface wave-field twin — legoESM vs NEMO, sampled every model step

**The question.** Do the first-days sea-surface wave patterns match between the
two models? Both start from the same NEMO day-180 restart and single-step
comparisons match closely, but the propagating free-surface field had never
been compared at its own timescale: the saved twin output was DAILY (32 model
steps) while a barotropic gravity wave crosses the DINO basin in about four
hours.

**The short answer.** The wave patterns match. Where a wave actually exists,
legoESM reproduces NEMO's free-surface response to within a few per cent, with
the spectral peaks at identical frequencies and no band carrying excess
legoESM energy. Getting to that answer required a second experiment, because
the free run turned out to have no waves in it at all.

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

**2. The time-level identity I first wrote was wrong, and its residual was
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
- **legoESM's free surface carries a two-step computational mode the oracle
  very nearly does not.** Measuring the alternating component directly over the
  first eight steps: NEMO 3.9e-7 m, legoESM 7.0e-6 m — **18× the oracle** —
  decaying to 1.4× over the second half of the run. In absolute terms this is
  small (7e-6 m on an 0.83 m field), and it is a start-up transient rather
  than a growing mode. But it is present in the free run and ABSENT from the
  impulse run (where the ratio is 1.07×), which places it in how legoESM
  settles a balanced background, not in how it propagates a wave.
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

**The two models propagate it at the same speed.** Cell-to-cell phase lag is
unresolvable here, but the RADIUS the response has spread to is not, because
the wave covers about nine cells per step. The energy-weighted mean radius of
the response, legoESM over NEMO, has a **median of 1.0013 over even samples**
(1.0020 over all): the spreading rates agree to about a tenth of a per cent.
The odd samples run 0.72–0.90 and the even ones 0.98–1.02, which is the
leapfrog alternation described below rather than a speed difference — it
cancels on alternate samples, a speed error would not.

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
1.369e-4 m — **legoESM 1.07× the oracle** — and over the second half of the run
both are 8.51e-7 m, a ratio of 1.00. When a genuine wave dominates the
signal, legoESM's computational mode is not under-damped.

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
- **IMPULSE lane: (i) PASSES at 0.00003×, and that pass is VACUOUS** — the
  "one-step floor" in this lane IS the peak of the response error, which then
  decays, so 160 × the floor cannot be exceeded by construction. The criterion
  was written for a lane where the error grows. **(ii) PASSES with margin** —
  worst ratio 1.17 of an allowed 2.0, across 11–21 live bands per probe, and
  that is the criterion the verdict rests on.

**CONFIRMED: the sea-surface wave patterns match.** The claim rests on the
impulse lane, which is the only one with a wave in it. It does NOT rest on the
free lane, whose registered pass is an artefact of there being no signal.

**The feedback-tier finding is the free lane's**, and it is separate from the
wave question: in a free run legoESM's free surface carries a two-step
computational mode 18× the oracle's over the first eight steps (decaying to
1.4×), and the step-to-step excess is enriched 7.5× on land-adjacent cells,
peaking at ~58× on one cell of the eastern wall. The same mode is only 1.07× the
oracle's when the impulse gives it a real wave to carry. This is the locus the
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
- **Instrument tests.** 50 direct unit tests, written as known-answer and
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

- Nothing about barotropic wave PHASE SPEED. The sampling cannot resolve
  cell-to-cell lag in this basin (0.11 of a step per cell) and no output either
  model currently writes can. That would need the free surface inside the
  barotropic substep loop, at ~90 s.
- Nothing about the 78–435× kick-amplification result that motivated the run.
  That measures the growth of an injected perturbation over thirty days; this
  measures the difference between two models over five.
- Nothing below ~2e-5 m in the free lane, which is where NEMO's own two time
  levels sit apart.
