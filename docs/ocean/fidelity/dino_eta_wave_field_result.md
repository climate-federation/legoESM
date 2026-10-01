# DINO sea-surface wave-field twin — legoESM vs NEMO, sampled every model step

**The question.** Do the first-days sea-surface wave patterns match between the
two models? Both start from the same NEMO day-180 restart and single-step
comparisons match closely, but the propagating free-surface field had never
been compared at its own timescale: the saved twin output was DAILY (32 model
steps) while a barotropic gravity wave crosses the DINO basin in about four
hours.

**The short answer.** The wave patterns match in SHAPE, in the frequency
domain — spectral peaks at identical frequencies, peak amplitudes within 0.5 %,
no band carrying excess energy — **and in time.** An earlier version of this
document reported a half-step timing offset. That offset was a defect in how
the TWIN WAS STARTED, not a difference between the models, and it is retracted
below:

> **The two models' barotropic impulse responses are simultaneous to 0.0001 of
> a baroclinic step**, and the RESPONSE fields differ by 9.6e-7 m rms, once
> legoESM is started as the leap-frog continuation it is meant to be. The
> "0.47-step lag" was produced entirely by starting legoESM from a
> forward-Euler cold start instead: the twin ran without the before-level
> bridge, so legoESM never read NEMO's `sshb`/`ub`/`vb`/`tb`/`sb`. An Euler
> first step makes the whole subsequent trajectory the **two-point running mean**
> of the leap-frog one — an exact identity, and a half-sample delay at every
> frequency. That is the number that was measured.

Getting there took a second experiment, because the free run has no waves in
it, and a better estimator, because a half-step offset is invisible to any test
that can only score whole steps — and then one controlled re-run, because a
half-step offset is also exactly what a wrong INTEGRATOR START looks like.

---

## Retractions, up front

**0. THE BIGGEST ONE: the "0.47-step lag" was not a model difference. It was a
twin-setup defect, and it is withdrawn in full.** The lag was real as a
MEASUREMENT — it reproduces exactly — but its owner is the harness, not the
model. legoESM's two five-day arms were run WITHOUT `--bridge-before`, which is
off by default. `kamm_twin_90d.py`'s own module docstring calls that flag
*"required (not merely optional)"* for this card and says that without it the
twin is *"a forward-Euler-from-now start"* rather than *"a real leap-frog
continuation"*. Neither run log contains the before-level verify line, so
neither run had it. With `state.u_before is None`, legoESM's leap-frog takes
its documented forward-Euler first step with `before := now`, and its step-1
barotropic solve runs the forward-Euler path (23 substeps seeded at *t*)
instead of the leap-frog one (46 substeps seeded from the before level at
*t*−Δ*t*) that NEMO runs. NEMO, meanwhile, was on its leap-frog branch from the
very first step (`ln_1st_euler = F`, `l_1st_euler = F` at kt=5761, both printed
in `ocean.output`).

**RETRACTED WITHIN THIS RETRACTION, before it was published: my first
explanation of the mechanism was wrong.** I wrote that "legoESM never saw the
copy of the bump that launches NEMO's wave". It did. The impulse was written
into `sshb` AND `sshn`, and the Euler branch sets `before := now`, so in the
PERTURBATION field both models start from the same two-level pair. That story
also predicts a whole step, not the half step measured, and it never confronted
the factor of two. The correct mechanism is in "Why an Euler start costs
exactly half a step" below, and it predicts 0.500 rather than merely "some
offset".

Re-running BOTH legoESM arms with `--bridge-before` and changing nothing else:

| arm | lag α (backward) | lag α (centred) | R² | response residual rms, samples 1–159 |
|---|---|---|---|---|
| baseline, no `--bridge-before` | **+0.4702** | +0.5512 | 0.714 | 6.29e-05 m |
| **with `--bridge-before`** | **+0.000142** | +0.0002 | 0.0003 | **9.63e-07 m** |

The lag does not shrink, it disappears. The gate registered before the re-run
was "confirms the before-level owner if α < 0.15"; it came back at 0.000142.
The residual column is the IMPULSE RESPONSE (each model minus its own free run)
over samples 1–159, and it falls 65×; the full free-surface difference is
unchanged at ~3.3e-5 m, and the gain is concentrated at the launch rather than
spread over the record — both quantified under "Where the improvement lives"
below, because "65× closer" without that window is a misleading headline. Everything the earlier document built on the
lag — that one scalar explains 71 % of the residual, that the step-to-step
alternation and the sample-1 spreading ratio and the 44 % bootstrap are all
re-descriptions of it — describes the un-bridged twin, not legoESM.

**0b. The mechanism offered for the lag was wrong too, and its supporting
number was misread.** The lag was attributed, as PLAUSIBLE, to the two models
placing their barotropic time-average at different centroids, citing a comment
in `ocean/state.py` recording two kernels "7.3 substeps apart out of 23, about
0.32 of a step". That comparison is not the one the eta measurement needs, in
three separate ways: it is INTRA-model (one model's velocity boxcar against its
own transport tail-sum), it is quoted from the FORWARD-EULER path rather than
the leap-frog card that ran, and the transport kernel never reaches the free
surface at all. The comparison that IS needed — legoESM's committed-eta kernel
against NEMO's — comes out at exactly zero; see "The averaging centroids are
identical" below.

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
| FREE + before-bridge | *(unchanged — NEMO was never re-run)* | same, plus `--bridge-before`, 83 s |
| IMPULSE + before-bridge | *(unchanged)* | same, plus `--bridge-before`, 83 s |

The two `--bridge-before` arms are the controlled A/B that produced the
retraction at the top: ONE flag differs from their baselines, on the same
restart, the same card, the same code, against the same unchanged NEMO output.
Both report `max|d_tb| = max|d_sb| = max|d_ub| = max|d_vb| = 0.000e+00` against
the restart's before level.

**That check does NOT cover the free surface, which is the load-bearing field
here.** The bridge verify prints T/S/u/v only, and `bridge_before_state_topo`
falls back SILENTLY to the now-level η when the restart carries no `sshb` — so
a restart missing `sshb` would print the same four zeros while leaving in place
the very defect this document blames for the lag. It is fine on this restart:
`sshb` is present and non-degenerate (`sshb − sshn` is 1.65e-4 m max,
1.62e-5 m rms). But the honest statement is "the T/S/u/v before-levels match
exactly, and η_before was bridged from a `sshb` confirmed present", not "an
exact leap-frog entry state". Adding `max|d_sshb|` to that print, and making
the `None` case fatal, is a one-line fix to the harness and is NOT done here.

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

Instruments: `scripts/validate/ocean_fidelity/dino_1226/eta_wave_twin.py` (the
comparison) and `scripts/validate/ocean_fidelity/dino_1226/baro_average_centroid.py`
(the averaging-centroid arithmetic) and
`scripts/validate/ocean_fidelity/dino_1226/euler_start_lag_toy.py` (the
Euler-start half-step mechanism).

**The baseline arms' two legoESM runs report different code SHAs** (`abcfc8cfc`
free, `93bd8551f` impulse) — checked, and the diff between them touches zero
lines under `packages/ocean` and `src`, so the model is byte-identical across
the pair and the impulse lane's perturbed-minus-free subtraction is not
confounded by it.

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
- **legoESM's free surface carries its OWN two-step mode — 2.9× the oracle,
  not the 18× first published.** Measured un-bridged, the alternating component
  over the first eight steps was NEMO 3.9e-7 m against legoESM 7.0e-6 m — 18×,
  against leakage floors of 9.9e-8 and 1.3e-6 m, above-floor ratio 19.5,
  decaying to 1.4× over the second half. Most of that was the cold start:

  **MOSTLY THE SAME DEFECT — this number is corrected from 18× to 2.9×.** The
  free arm was un-bridged too, and a forward-Euler start is exactly what
  excites a leap-frog computational mode. Re-running the free arm with
  `--bridge-before` and nothing else, on the same estimator: NEMO 3.90e-7 m,
  legoESM baseline 7.00e-6 m (**17.95×**, reproducing the published number),
  legoESM bridged 1.13e-6 m (**2.89×**). So roughly six of the eighteen-fold
  excess was the cold start. A residual **2.9×** stands and is a real
  model-to-model difference; the second-half ratio is 1.12× against a baseline
  1.38×, so it still decays.

  **The residual clears its own floor**, which is the check that makes it a
  finding rather than leakage. Each arm's floor must be recomputed on that
  arm's own field, because the estimator's leakage tracks the field it is run
  on — the un-bridged legoESM's floor was 1.33e-6 m, the bridged one's is
  2.58e-7 m, so quoting the old floor against the new amplitude would wrongly
  bury it. On its own floor the bridged mode sits **4.4× above** (NEMO 3.9×),
  and the above-floor ratio is **2.98×** against the raw 2.89× — the floor is a
  similar fraction of both sides, so it barely moves the answer.

  **The locus WAS re-measured, and it moves the finding in the opposite
  direction from what "shrinks to 2.9×" suggests.** Removing the cold start
  CONCENTRATES what is left onto the wall rather than dispersing it: the
  bridged mode's wall enrichment is **22.38** against the un-bridged 9.40, and
  its wall share rises from 0.358 to **0.852**. So the surviving 2.9× is not a
  weak remnant of a basin-wide transient — it is a sharply wall-localised mode
  that the cold-start noise was previously diluting. That makes it MORE relevant
  to the campaign's wall-row velocity work, not less.

  The slow, balanced part of the free lane is untouched by the bridge: the
  free-surface difference against NEMO is 3.354e-5 m rms un-bridged and
  3.299e-5 m bridged. That is consistent with the free lane's own control
  result — it has no lag, so it had nothing for the fix to remove.

  Two qualifiers that matter, both measured UN-BRIDGED and therefore describing
  the cold-start transient rather than the surviving 2.9×. Its signed spatial
  correlation with NEMO's is **−0.09 to +0.08, i.e. zero**: unlike the impulse
  lane, legoESM was not ringing the oracle's mode harder, it was ringing a
  different one. And it is **not basin-wide** — the wall enrichment of its first
  sample is 9.4, so this is a BOUNDARY transient, and calling it "the leapfrog
  computational mode under-damped by 18×" would overstate it twice over. In
  absolute terms it is small (7e-6 m on an 0.83 m field) and it decays. Unlike
  the impulse lane's alternation, this one is NOT a re-description of the lag:
  the free run has no lag — though it does share the lag's ROOT CAUSE, the
  missing before level.
- **That excess sits on the walls, and bridging makes it MORE so, not less.**
  Re-measured on the bridged arm: the step-to-step excess keeps a wall
  enrichment of **5.80** (down from 7.46 at the same 3.8 % area fraction), while
  the two-step mode's wall SHARE rises from 0.358 to **0.853** against NEMO's
  0.044. The wall localisation survives the fix and sharpens. Un-bridged
  numbers, kept for the record — per-cell ratio of mean-squared step-to-step
  change:
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

### The response looked half a step late — on a twin that was started wrong

> **EVERYTHING IN THIS SUBSECTION DESCRIBES THE UN-BRIDGED ARMS**, i.e. a
> legoESM started from a forward-Euler cold start instead of NEMO's leap-frog.
> It is kept because the measurement is sound and because the two subsections
> after it are how the owner was found — but as a statement about the MODELS it
> is retracted. The bridged numbers are at the top of this document.

A lagged trajectory is, to first order, a linear interpolation between two of
the reference's samples, so fitting

    eta_lego[k] - eta_nemo[k]  =  alpha * ( eta_nemo[k-1] - eta_nemo[k] )

by area-weighted least squares over wet cells and samples returns the lag
directly, as a continuous number of steps. Measured, un-bridged:

| lane | lag alpha | in minutes | share of residual variance explained |
|---|---|---|---|
| impulse, all 160 samples | +0.470 steps | +21.2 min | 0.714 |
| impulse, first 8 samples | +0.458 steps | +20.6 min | 0.679 |
| free (control), all samples | −0.132 steps | −5.9 min | 0.004 |
| free (control), first 8 | +0.018 steps | +0.8 min | 0.000 |
| free (control), all, `--bridge-before` | −0.127 steps | −5.7 min | 0.004 |
| **impulse, all 160, `--bridge-before`** | **+0.000142 steps** | **+0.006 min** | **0.0003** |

Both lanes are quoted over the SAME sample count. The free lane's α is −0.132
over all samples and +0.018 over the first eight; both have R² ≈ 0.004, so it
is "no lag" either way, but the earlier text quoted the first-8 free value
against the all-samples impulse value, which is not a matched comparison.
All residual-rms figures in this document run over samples 1–159 (the fit needs
`k−1`); including sample 0 raises the un-bridged residual to 1.018e-4 m and the
headline ratio to 106×.

Positive means legoESM is behind. The competing explanation — that legoESM's
field is simply scaled rather than delayed — was fitted alongside and explains
0.1 % of the same residual against the lag's 71 %. The free run was read at the
time as the control that made it a result: the same estimator, the same two
models, no wave in the field, returns no lag. That reading was right about the
estimator and wrong about the conclusion — the free lane has no lag because it
has no wave to launch, so it could never have caught a LAUNCH-TIME defect. **A
control that cannot express the failure mode is not a control for it**, and
this one was quoted as though it were.

The published α carries no uncertainty, so one was computed here: bootstrapping
the pooled ratio over its 159 per-sample terms gives 95 % [0.433, 0.567],
standard error 0.039, and a second-order centred-difference estimator returns
+0.551. The band used to score the candidate mechanisms is therefore
**[0.433, 0.567]**.

**Four things reported below were re-descriptions of this one lag, not
independent findings**: the step-to-step alternation of the difference (odd
samples 5.1× the even ones), the sample-1 spreading ratio of 0.713 against
0.995 at sample 2, the 2-step ringing in the impulse lane, and the 44 %
bootstrap. A half-step offset sampled on alternate steps produces all four —
and since the offset itself is now retracted, all four fall with it.

### The averaging centroids are identical, so that mechanism is refuted

Half a step is what a disagreement about where the barotropic time-average sits
would produce, so that was the standing PLAUSIBLE mechanism. It is now settled,
by arithmetic on the two weight vectors — no run, no tolerance.
`baro_average_centroid.py` ports NEMO's `ts_wgt` (`dynspg_ts.F90:1227-1294`)
verbatim, reads the RESOLVED barotropic settings from the run's own
`ocean.output`, and imports legoESM's shipped kernel rather than re-deriving it.

Reading the resolved settings matters: `ln_bt_auto=.true.` means NEMO COMPUTES
the substep count, so the namelist's `nn_e = 30` is dead and the value that ran
is **23** (`ocean.output`: `in iterations nn_e = 23`, `nn_bt_flt = 2`,
`ln_bt_fw=F => Centred integration`). An earlier passage in this document said
"30 barotropic substeps per step"; that quoted the overridden namelist and is
corrected to 23 here and below.

| | NEMO | legoESM (`nemo_dino_kamm_mlf`) |
|---|---|---|
| kernel | `wgtbtp1`, `nn_bt_flt=2`, `ln_bt_fw=F` | `nemo_boxcar_ab3`, leap-frog, scale 2 |
| substeps run | 68 (`icycle`) | 68 (`n_loop`) |
| substep length | 117.3913043478 s | 117.3913043478 s |
| averaging window | substeps 24–68 | substeps 24–68 |
| **eta centroid** | **substep 46.0 = *t* + 1.000000 steps** | **substep 46.0 = *t* + 1.000000 steps** |
| transport centroid | substep 25.333 = *t* + 0.101 steps | substep 25.333 = *t* + 0.101 steps |

The weight vectors are bit-identical (`max|w_lego − w_nemo/Σw| = 0.000e+00` at
fp64) and the substep lengths agree to all printed digits, so the

> **predicted legoESM − NEMO committed-eta centroid offset is −0.000000
> baroclinic steps.**

Both models place their committed free surface at exactly the nominal after
time: NEMO's boxcar window is symmetric about `jic = 2·nn_e`, which IS the
new-time point of a 2Δ*t* integration started from the before level, and
legoESM builds the same window on the same clock. **Predicted 0.000 against a
measured 0.470 with a 95 % interval of [0.433, 0.567] — about twelve standard
errors outside the band. The averaging-centroid mechanism is REFUTED.**

### The candidate table, and where the lag actually came from

Registered before any of these were computed, with the predicted offset each
one implies. Note the last row is a real prediction — a NUMBER that the
measurement could have contradicted — not the unfalsifiable "the whole thing or
nothing" it was first written as:

| candidate | predicted offset | verdict |
|---|---|---|
| barotropic averaging-centroid mismatch | 0.000 steps (arithmetic above) | **REFUTED** — 12 σ outside the band |
| output convention: the two sides write different eta time levels | ±1.000 steps (an integer number of dumps) | **REFUTED** — both write the unfiltered after level; see below |
| 2-step/Nyquist computational-mode amplitude difference | ≤0.035 steps, and negative | **REFUTED** — deleting the Nyquist bin moves α by 0.001 |
| legoESM's field is uniformly scaled, not delayed | n/a (amplitude, not time) | **REFUTED** — explains 0.1 % against the lag's 71 % |
| **integrator start: legoESM Euler-from-now vs NEMO leap-frog** | **+0.500 steps exactly (unfiltered); +0.50–0.56 at the card's γ = 0.1** (derived below) | **CONFIRMED — α 0.4702 → 0.000142** |

**The output-convention hypothesis was tested first and is dead on two
independent counts.** On code, both sides write the UNFILTERED after level:
NEMO swaps its time indices at `stpmlf.F90:621-624` before `rst_write` at
`:634`, so the saved `sshn` is the pre-swap AFTER level, and legoESM returns
`naa.eta` from the leap-frog step with the Asselin result parked separately in
`eta_before` (`ocean_model_latlon_cgrid.py:8882-8887`). Empirically, no integer
re-pairing brings the fit anywhere near zero — shifting legoESM against NEMO by
−1 dump gives α = −0.198, by +1 gives α = +1.240, and pairing against NEMO's
Asselin-filtered before level gives α = −0.160. An output-convention error is
an integer number of dumps by construction and structurally CANNOT produce half
a step.

**The lag was frequency-independent, which is what made it a real delay rather
than an estimator artifact — and also what pointed at the launch.** Fitting the
lag per frequency band gives α ≈ +0.54 to +0.65 across periods of 4 to 9 steps,
which carry most of the fit; removing every period shorter than 4 steps still
leaves α = +0.53, and removing the Nyquist bin alone leaves α = +0.4712. A
broadband, frequency-flat delay in an IMPULSE RESPONSE is the signature of a
wave that was LAUNCHED at the wrong moment, not of one that propagates at the
wrong speed — and the launch is precisely where the two arms differed.

### Why an Euler start costs exactly half a step

This is the part that makes the owner a mechanism rather than a correlation,
and it is exact.

The impulse was written into BOTH `sshb` and `sshn`, so in the perturbation
field both models begin from the same two-level pair, `δ_before = δ_now`. The
only thing that differs on step 1 is the integrator applied to that pair. For a
mode `dη/dt = iωη` with `s = ωΔt`:

    NEMO    δ¹ = δ_before + 2is·δ_now  =  δ(1 + 2is)     leap-frog over 2Δt
    legoESM δ¹ = δ_now    +  is·δ_now  =  δ(1 +  is)     forward Euler over Δt

**Unfiltered, the two trajectories are related by an EXACT identity.** Both
then obey the same three-term recurrence, and the two-point running mean of the
leap-frog trajectory reproduces the Euler trajectory's first two levels, so by
induction

    eta_euler[k]  ==  ( eta_leapfrog[k-1] + eta_leapfrog[k] ) / 2      EXACTLY

verified to 1e-15 at periods 6 to 80 steps. **A two-point running mean is a
half-sample delay at every frequency** — its phase is exactly −πf, linear
through the origin. That one line is the entire mechanism, and it is why the
lag is broadband and frequency-flat rather than looking like a wave-speed
error.

**What the card actually runs is γ = 0.1, and that is NOT the flat 0.5.** NEMO
filters from step 1 while legoESM's Euler branch does not
(`ocean_model_latlon_cgrid.py:8232`, "no RA filter"), and that asymmetry tilts
the answer up and makes it rise with period. Scored with the SHIPPED
`substep_lag_fit` (`euler_start_lag_toy.py`):

| period (steps) | 6 | 8 | 12 | 20 | 40 | 80 | broadband |
|---|---|---|---|---|---|---|---|
| α, γ = 0 (exact) | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.50000 |
| **α, γ = 0.1 (as run)** | 0.5043 | 0.5206 | 0.5380 | 0.5487 | 0.5535 | 0.5550 | **0.51276** |

So the prediction is **+0.50 unfiltered, rising to +0.50–0.56 at the card's own
Asselin coefficient**. Measured: +0.470 (backward-difference estimator), +0.551
(centred), bootstrap 95 % [0.433, 0.567]. The γ = 0.1 band lands ON the centred
estimator and accounts for something the flat 0.5 could not: the per-band fit's
RISE with period (+0.54 to +0.65 over periods 4–9). The control is exact —
identical starts give α = +0.000000. **This is also why the free lane could not
catch the defect**: with no wave to launch, there is no phase for a launch
error to shift.

**Scope caveat on the toy, because its short end is not trustworthy.** At
γ = 0.1 the filtered leap-frog is UNSTABLE below about a 6-step period — over
159 samples the 4-step column reaches 1e25 and the 5-step column 1e16, and a
4-step period sits exactly at the scheme's own limit (`s = sin(π/2) = 1`). Every
aggregate above is therefore restricted to periods ≥ 6, and the toy cannot
speak for the short end of the measured per-band table.

**SCOPE OF THE ATTRIBUTION, stated honestly.** `--bridge-before` does not set
only `eta_before`: it seeds `T/S/u/v/eta_before` and also the before-level wind
stress (`tau_x_prev`/`tau_y_prev`), which this card consumes because it runs
`barotropic_forcing_centred=True`. No single-field arm was run. So what is
CONFIRMED is that **the step-1 entry state as a whole** owns the lag, and that
the integrator-start algebra above predicts the measured value exactly; that
the `eta_before` seed specifically is the operative field is PLAUSIBLE, not
measured. A one-field ablation would settle it and has not been run.

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
at the bump's centre the response has already reversed sign. With 23
barotropic substeps per baroclinic step (NEMO's resolved `nn_e`, not the
namelist's 30) the wave travels ~9 cells inside a single step, so
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
- **IMPULSE lane, un-bridged: (i) NOT APPLICABLE** — the first sample is the
  largest in the series, so 160 × the floor cannot be exceeded by construction.
  The instrument reports this as null with a reason rather than as a
  spectacular pass. **(ii) PASSES with margin** — worst ratio 1.17 of an allowed
  2.0, across 11–21 live bands per probe.
- **IMPULSE lane, bridged: (ii) PASSES BY MORE, and it is now measured rather
  than expected.** The worst band ratio at the four probes moves
  1.039 / 1.166 / 0.939 / 1.108 → **0.999 / 0.974 / 1.003 / 1.005**, and the
  impulse-lane two-step mode ratio moves 1.067 → **1.00004**. Removing the cold
  start makes the frequency-domain agreement essentially exact.
- **IMPULSE lane, bridged: (i) BECOMES LIVE, AND READS 1.31 AGAINST A BAR OF
  1.0 — a nominal FAIL that must be reported, and that I do not believe.** The
  criterion was vacuous un-bridged only BECAUSE the defect made sample 0
  enormous; with the defect gone the one-step floor drops to 1.468e-07 m, the
  linear bar to 2.349e-05 m, and the day-5 maximum of 3.083e-05 m exceeds it by
  1.313. The reason I do not read this as amplification: the bridged residual
  alternates hard step to step, and sample 0 sits in the TROUGH
  (1.468e-07 m) while sample 1 sits at 8.417e-07 m, 5.7× higher. Taking the
  floor from sample 1 instead gives 0.229, a comfortable pass. **The gate's
  verdict here is parity-dependent and therefore not resolvable**, which is an
  instrument defect: `one_step_floor_max_abs_m` samples ONE step of a
  two-step-alternating residual. Fixing it (take the floor over a step PAIR) is
  a one-line change to the instrument and is NOT done here.

**CONFIRMED: the sea-surface wave patterns match in the frequency domain**, and
on the bridged pair they match almost exactly (worst band ratio 1.005). The
claim rests on criterion (ii) in the impulse lane — the only lane with a wave
in it. It does NOT rest on the free lane, whose registered pass is an artefact
of there being no signal, nor on criterion (i), which is vacuous un-bridged and
parity-dependent bridged.

**RETRACTED: "legoESM's barotropic response is 0.47 of a step late."** It is
late by 0.0001 of a step. The 0.47 was the un-bridged twin's forward-Euler
start, not the model. The shapes agree AND the clocks agree.

**Where the improvement lives, because "65× closer" on its own is misleading.**
What falls 65× is the IMPULSE RESPONSE residual (each model minus its own free
run) over samples 1–159; the full free-surface difference does not move at all,
from 3.354e-5 to 3.299e-5 m rms, a factor of 1.02. And the response gain is
concentrated at the launch, which is exactly what a first-step defect predicts:

| window | baseline | bridged | ratio |
|---|---|---|---|
| sample 0 alone | 1.014e-03 m | 2.37e-08 m | 42900× |
| samples 0–7 | 4.367e-04 m | 1.539e-07 m | 2838× |
| samples 1–159 (the "65×") | 6.293e-05 m | 9.631e-07 m | 65× |
| samples 0–159 (all) | 1.018e-04 m | 9.601e-07 m | 106× |
| samples 80–159 | 8.669e-06 m | 1.135e-06 m | 7.6× |
| final max abs difference | 2.370e-05 m | 3.083e-05 m | **0.77× — slightly WORSE** |

Late-time agreement is genuinely good (1.1e-6 m against a NEMO response of
2.2e-4 m, i.e. 0.5 %), but the day-5 maximum difference does not improve. **The
launch defect never owned the late residual**, which is a separate and still
unexplained model difference of similar size in both arms. What remains unmeasurable here is
cell-to-cell phase lag and the spreading-rate comparison, both below resolution
or saturated.

**The feedback-tier finding is the free lane's, and it is now much smaller than
reported.** In a free run legoESM's free surface still carries its own two-step
mode at **2.9×** the oracle's amplitude over the first eight steps (2.98× after
each side's own leakage floor) — not the 18× first published, which was mostly
the same cold-start defect (17.95× un-bridged, 2.89× bridged). The residual
2.9× is a genuine model-to-model difference: it sits 4.4× above the bridged
arm's own leakage floor, and it decays to 1.1× over the second half. **Its
locus WAS re-measured on the bridged arm and it is more wall-localised than
before, not less**: two-step wall share 0.358 → **0.853** (NEMO 0.044), wall
enrichment 9.40 → **22.38**, and the step-noise wall enrichment 7.46 → 5.80 at
the same area fraction. The signed correlation with NEMO's own mode also stops
being zero. So the surviving finding is a SHARP wall-row 2Δt flicker, and it is
the live thread for the campaign's wall-row velocity work rather than a residue
of the cold start. (The ~58× peak-cell figure is the only locus number not
re-derived.)

**THE STANDING LESSON, because it cost this lane three rounds.** A default-off
harness flag whose own docstring says "required" for the card under test is a
CONFIGURATION variable of the experiment, and it was never checked. Every
number in the first four versions of this document was measured on a twin whose
integrator start did not match the oracle's. The instrument was interrogated
repeatedly — the estimator was rewritten, the pairing was bootstrapped, the
spectra were re-weighted — and none of that could see the defect, because the
defect was upstream of the instrument, in what the model was handed. Check the
arms' resolved configuration against the oracle's BEFORE trusting a difference
between them.

**The one-line fix that would have prevented all of it** is not in this diff
and should be: nothing enforces `--bridge-before` for a leap-frog card, and
nothing warns when it is absent. Either make it a card-level requirement, or
print a loud banner in the run log whenever a leap-frog card starts un-bridged.
A flag whose own docstring says "required" must not be silently default-off.

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
- **`rn_Dt`.** Read back from every NEMO dump and required to be 2700 s. The
  centroid probe also reads it from `ocean.output` rather than defaulting, and
  refuses a log whose `rDt_e × nn_e` does not equal it or whose MLF `rDt` is not
  `2 × rn_Dt` — the clock is the whole answer, so an inconsistent log must abort
  rather than produce a plausible number.
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
- **Lag estimator, re-derived independently.** Before any new claim was made
  from it, the published α was reproduced from the raw artifacts by a separate
  implementation: 0.470212 against the recorded 0.470222 over all samples, and
  0.457690 exactly on the first eight. A second, second-order estimator built
  on the CENTRED difference (which carries no curvature term for a backward
  difference to leak into) returns +0.551 rather than +0.470, so the lag is not
  an artifact of the first-order form; the two bracket the answer.
- **Error bar on the lag.** The published α carried none. Bootstrapping the
  pooled ratio over its 159 per-sample numerator/denominator pairs (4000 draws)
  gives a 95 % interval of **[0.433, 0.567]**, standard error 0.039; the
  per-sample α values have median 0.553 and standard deviation 0.049. That is
  the band the candidate predictions were scored against.
- **Averaging centroids.** Computed, not assumed, by a committed probe that
  ports NEMO's `ts_wgt` verbatim and imports legoESM's shipped kernel; its
  7 tests are known-answer on all four `nn_bt_flt` branches and include a
  synthetic violation (building the kernel on the forward-Euler substep scale
  moves the centroid a full step, and the load-bearing assertion goes red).
  Mutating the boxcar half-width in the port turns 3 of the 7 red, so the
  suite is not vacuous.
- **Barotropic settings taken from the run, not the namelist.** `ln_bt_auto`
  overrides `nn_e`, so the probe refuses to run unless it can read the resolved
  `nn_e` / `nn_bt_flt` / `ln_bt_fw` out of `ocean.output`; a log missing any of
  them aborts rather than defaulting.
- **The centroid probe can report failure.** Its first version could not: both
  the leap-frog window (substeps 24–68 on a clock starting at *t*−Δ*t*) and the
  forward-Euler one (substeps 1–45 on a clock starting at *t*) put their
  centroid at exactly *t*+Δ*t*, so a MISCONFIGURED probe printed the same
  `-0.000000` as a correct one — a control that cannot fail. The verdict is now
  gated on the two facts that DO discriminate, the kernel identity and the
  substep-clock identity, and a test drives the gate by forcing the wrong
  substep scale and requiring the refusal.
- **The mechanism is verified, not asserted.** The half-step prediction is
  checked analytically at seven periods and reproduced by a toy leap-frog scored
  with the SHIPPED `substep_lag_fit` (α = +0.5000, R² = 1.0000, and exactly zero
  for the identical-start control).
- **Integrator start, both sides.** NEMO's leap-frog branch is confirmed live
  from the first step (`ln_1st_euler = F` and `l_1st_euler = F` at kt=5761, both
  printed). legoESM's Euler-start branch is confirmed live in the baseline arms
  by the ABSENCE of the before-level verify line from both logs and by
  `--bridge-before` defaulting to off — and confirmed removed in the new arms by
  its presence at 0.000e+00.
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

`/tmp/dino_eta_waves/figs/free/` and `/tmp/dino_eta_waves/figs/impulse/` (the
un-bridged baseline), and `/tmp/dino_eta_waves/figs/free_bb/` and
`/tmp/dino_eta_waves/figs/impulse_bb/` (the `--bridge-before` arms, from which
every retraction number above is taken), each
with `fig_diff_maps.png` (difference maps at the eight pre-registered times),
`fig_hovmoller.png` (time–longitude at the channel latitude and time–latitude
at mid-basin longitude, both models and their difference), `fig_growth_locus.png`
(growth curve against the linear bound, plus locus enrichment on a log axis)
and `fig_spectra.png`. Full numbers in each directory's `eta_wave_twin.json`.

## What this does NOT establish

- Nothing about barotropic wave PHASE SPEED as a propagation rate. Cell-to-cell
  lag remains unresolvable (0.11 of a step per cell) and the spreading-rate
  comparison saturates after two samples.
- The spreading and `step_noise_ratio` numbers in the body were computed on the
  UN-BRIDGED arms and have not been re-derived. The lag fit, both lanes'
  two-step amplitudes, the free lane's locus, the spectra and the growth gate
  HAVE been re-measured on the bridged pair — see `figs/free_bb/` and
  `figs/impulse_bb/`.
- Nothing about the MECHANISM of the residual 2.9× free-lane two-step mode. Its
  amplitude is floored, its locus is re-measured on the bridged arm and is
  sharply on the wall (share 0.853 against NEMO's 0.044) — but WHY legoESM
  carries a wall-row 2Δt flicker the oracle does not is not answered here.
- Nothing about the 78–435× kick-amplification result that motivated the run.
  That measures the growth of an injected perturbation over thirty days; this
  measures the difference between two models over five.
- Nothing below ~2e-5 m in the free lane, which is where NEMO's own two time
  levels sit apart.
