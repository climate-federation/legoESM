# DINO sea-surface wave-field twin — legoESM vs NEMO, sampled every model step

**Question asked.** Do the first-days sea-surface patterns match between the two
models? Both start from the same NEMO day-180 restart, single-step comparisons
match to small levels, and the 90-day climate is scored — but the propagating
free-surface field had never been compared at its own timescale. The saved twin
output was DAILY (32 model steps), while a barotropic gravity wave crosses the
DINO basin in roughly eight hours.

**What was run.** Five days from `DINO_00005760_restart.nc` (md5
`ad5ba426…`), 160 steps of 2700 s, sea surface height saved EVERY step on both
sides.

- NEMO: the certified binary `nemo.exe.certified_d3cf9242` (md5
  `d3cf9242289b633d671013d0299803a3`, byte-identical copy, original untouched),
  the `RUN_90D_TWIN` namelist with two fields changed — `nn_itend` 8640 → 5920
  and `nn_stock` 320 → 1 — 16 MPI ranks, `STOP 0`, no errors. Run directory
  `/tmp/dino_eta_waves/nemo_5d` (namelists and log retained; the 35 GB of raw
  per-step restarts were deleted after extraction and are reproducible in
  ~3 minutes).
- legoESM: `kamm_twin_90d.py nemo_dino_kamm_mlf --days 5 --output-every-steps 1`
  at fp64 on one GPU, 74 s. Day-0 gate: `max|dT| = max|d_eta| = max|du| =
  max|dv| = 0.000e+00` against the restart, so the two models start from
  bit-identical data.

Instrument: `scripts/validate/ocean_fidelity/dino_1226/eta_wave_twin.py`.

## Time centering — cited, then measured

NEMO swaps its time-level indices at `MY_SRC/stpmlf.F90:621-624` BEFORE calling
`rst_write(kstp, Nbb, Nnn)` at `:634`, and calls it without a `Kaa` argument, so
the MLF branch `restart.F90:197` writes `sshn` from the POST-swap `Kmm`. That
level is the pre-swap AFTER level, i.e. **the sea surface height after `kt`
steps** — and it is the UNFILTERED after-level; the Asselin-filtered level is
what becomes `sshb`. The same ordering is in the untouched upstream
`src/OCE/stpmlf.F90:402`, so this is core NEMO, not a MY_SRC edit.

The empirical discriminator agrees but only weakly (rms over 160 samples, wet
cells):

| pairing | rms difference |
|---|---|
| legoESM(N) vs NEMO `sshn`(N) — **the registered pairing** | 3.628e-05 m |
| legoESM(N) vs NEMO `sshb`(N) | 4.096e-05 m |
| legoESM(N) vs NEMO `sshn`(N-1) | 4.108e-05 m |
| legoESM(N-1) vs NEMO `sshn`(N) | 3.999e-05 m |
| NEMO's own `sshn`(N) − `sshb`(N) | 1.815e-05 m |

The registered pairing wins, but by only 10%, and NEMO's own two time levels
differ by half the model-to-model difference. **No conclusion below ~2e-5 m may
rest on the pairing choice.**

## The premise correction, which reframes the whole question

The mission assumed daily output was ALIASING an ~8 h barotropic wave. It is
not aliasing it — the wave is not in this field at all. NEMO runs a
split-explicit free surface with time-averaging over 30 barotropic substeps, so
the free surface handed to the baroclinic step is already the filtered, slow
one. Share of temporal variance over wet cells, 45-minute sampling:

| period band | NEMO | legoESM | their difference |
|---|---|---|---|
| < 6 h | 3.6e-06 | 4.2e-06 | 7.4e-03 |
| 6–24 h | 7.0e-05 | 8.2e-05 | 2.9e-02 |
| > 24 h | 0.99993 | 0.99991 | 0.9634 |

Both models put 99.99% of their free-surface variance at periods longer than a
day. The single-cell spectra say the same thing: at all four probe points the
spectral peak sits in the LOWEST resolved bin (period = the 120 h record
length), i.e. there is no wave peak to compare, only a trend.

A second, independent measurement says the same thing from the propagation
side. Along the ACC channel row (54.9°S, 51 adjacent wet pairs, 63.9 km cells)
the lag that maximises the cross-correlation of the sub-24 h free surface
between neighbouring cells is **exactly zero steps in BOTH models and in their
difference**, with 51 of 51 pairs inside one step. That is the arithmetic one
expects: a barotropic wave at sqrt(gH) = 210 m/s (H = 4506 m) crosses one
63.9 km cell in 304 s, which is 0.11 of a 2700 s time step, and the whole
3323 km basin in 4.4 h. Resolving that propagation needs a sampling interval
below about 300 s, roughly nine times finer than the model's own baroclinic
step. **No wave-speed or wave-phase claim — agreement OR disagreement — can be
made from baroclinic-step output for this basin.** The two models agree that
the resolvable signal is standing, which is the strongest statement the data
supports.

**Consequence:** this experiment CANNOT certify barotropic wave-speed or
wave-phase fidelity, because the observable does not contain barotropic waves.
Doing so would need the free surface INSIDE the barotropic substep loop, which
neither model currently writes at that cadence.

What the experiment CAN and does answer is the sharper question underneath it:
does legoESM inject free-surface structure on timescales the oracle does not
have, and where.

## Results

**The difference is small and SUBLINEAR.** Wet-cell max |eta_lego − eta_nemo|:

| target | actual t | max abs | rms | locus wall/eq/interior |
|---|---|---|---|---|
| 1 h | 0.75 h | 3.374e-05 m | 6.02e-06 m | 0.09 / 0.05 / 0.85 |
| 2 h | 2.25 h | 4.317e-05 m | 6.70e-06 m | 0.10 / 0.11 / 0.80 |
| 4 h | 3.75 h | 5.836e-05 m | 6.57e-06 m | 0.11 / 0.02 / 0.87 |
| 8 h | 8.25 h | 1.906e-04 m | 8.42e-06 m | 0.29 / 0.12 / 0.59 |
| 12 h | 12.00 h | 2.526e-04 m | 1.17e-05 m | 0.29 / 0.09 / 0.62 |
| 24 h | 24.00 h | 5.219e-04 m | 1.71e-05 m | 0.37 / 0.10 / 0.53 |
| 48 h | 48.00 h | 7.210e-04 m | 2.98e-05 m | 0.32 / 0.08 / 0.60 |
| 120 h | 120.00 h | 7.453e-04 m | 5.73e-05 m | 0.29 / 0.14 / 0.57 |

Targets are hours; `rn_Dt = 2700 s` makes an hourly cadence a non-integer number
of steps, so each target is realised at the nearest step and the true time is
printed beside it.

The maximum SATURATES near 8e-04 m around day 3 and then decays. Against the
pre-registered bound (160 × the one-step floor of 3.374e-05 m = 5.399e-03 m) the
day-5 value is **0.138×**, i.e. seven times below a merely linear accumulation
of the one-step disagreement. Relative to NEMO's own five-day change of the free
surface (max 2.85e-02 m, rms 2.77e-03 m), the day-5 difference is **2.6% of the
max and 2.1% of the rms**.

**But the one-step agreement is NOT roundoff, and that is worth naming.** After
a single step the two free surfaces differ by 3.374e-05 m where NEMO's own
one-step change is 2.193e-04 m — the free-surface INCREMENT disagrees by 19% at
the worst cell and 36% in rms. The certification "single-step comparisons match
to tiny levels" is true of the tendency terms; it is NOT true of the assembled
free-surface increment. That the accumulated difference nonetheless stays at 2%
of the signal means these per-step disagreements largely CANCEL rather than
accumulate.

**The excess energy is high-frequency and it lives on the walls.** The
difference field is ~500× less red than either model's field: 3.7% of its
variance sits at periods below 24 h versus 0.0074% for the signals. Per-cell
ratio of mean-squared step-to-step change, legoESM / NEMO:

- median **1.067** — the basin interior jitters alike,
- 90th percentile **3.48**,
- maximum **193.6**, at a land-adjacent cell on the eastern wall (46.7°N,
  49.5°E).

Splitting the EXCESS (legoESM minus NEMO step-to-step variance, clipped at zero)
by region: **wall 36.3%, equator 0.20%, interior 63.5%**, against area shares of
4.9% / 2.4% / 92.6% — an enrichment of **7.3× on the walls**, 0.7× in the
interior, 0.08× at the equator. legoESM's largest single-step free-surface jumps
(7.4e-04 m, 3.4× NEMO's largest anywhere) all occur at STEP 2 in the two
northern corners of the basin (69.2°N at both 1.5°E and 48.5°E) — a start-up
transient in the corners, absent in NEMO.

## Verdict against the pre-registration

Registered before the comparison ran: CONFIRM wave fidelity if (i) the
difference stays within 1.0 × (160 × the one-step floor) and (ii) no resolved
spectral band shows a power ratio beyond 2×.

- (i) **PASSES** — 0.138× the bound.
- (ii) **PASSES, but at the bar.** Worst per-band amplitude ratios at the four
  probe points: channel 1.49 (12 h band), equator 0.88 (30 h), west wall 1.67
  (12 h), mid-basin 1.92 (15 h). All four peaks agree in frequency exactly and
  in amplitude within 4%.

**The registered verdict is therefore CONFIRM — and it is worth less than it
looks**, because the observable turned out to contain no waves (see the premise
correction). The bar was written for a wave field and got applied to a slowly
evolving one.

**The finding that does carry weight** is the one the registered bar was not
written for: legoESM's free surface carries step-to-step energy the oracle does
not, concentrated 7.3× on the land-adjacent cells, peaking at ~200× on one
eastern-wall cell, with a start-up transient in the two northern corners at
step 2. This is the same locus the campaign's wall-row velocity work has been
circling, and it is a feedback-tier result rather than a certification.

## Controls that were run before any number above was believed

- **Mask.** The wet mask comes from NEMO's own `mesh_mask` surface `tmask`
  (9920 wet, 428 dry cells), never from "where the field is zero" — that would
  also delete genuinely zero ocean values and the equator. Every dry cell was
  then poisoned with 1e6 and the run ABORTS unless every masked statistic is
  bit-identical on the poisoned copy AND the unmasked statistic moves. It
  passed both halves.
- **Time level.** Checked in source (cited above), checked mechanically at
  extraction (consecutive dumps must differ; `sshn` must differ from `sshb`),
  and scored against three alternative pairings.
- **`rn_Dt`.** Read back from every NEMO dump and required to equal 2700 s, so
  the extractor cannot silently read a differently-configured run.
- **NaN.** Fatal everywhere; there is no `nanmax`/`nanmean`/`nansum` in the
  comparison path. Stitched fields must be fully finite or the extractor exits.
- **Equator.** Reported as its own locus region rather than normalised through;
  spectral ratios are formed only where NEMO carries real power (> 1e-3 of its
  peak), so a structural zero cannot manufacture a band mismatch.
- **Instrument tests.** 19 direct unit tests, built as known-answer and
  synthetic-violation controls rather than smoke tests: the spectrum recovers a
  0.37 m / 8 h sinusoid to 5%; a planted 2-step oscillation on the wall band is
  detected and attributed to `wall` with >99% of the excess; a motionless
  oracle cell is excluded rather than becoming an infinite ratio; removing the
  mask changes the statistic by 2000×.

## Figures

`/tmp/dino_eta_waves/figs/` — `fig_diff_maps.png` (difference maps at the eight
pre-registered times), `fig_hovmoller.png` (time-longitude at the channel
latitude and time-latitude at mid-basin longitude, both models and their
difference), `fig_growth_locus.png` (growth curve against the linear bound, plus
the locus shares), `fig_spectra.png` (spectra at the four probe points). Full
numbers in `eta_wave_twin.json`.

## What this does NOT establish

- Nothing about barotropic wave speed or phase. The observable does not contain
  those waves; see the premise correction.
- Nothing about the 78–435× kick-amplification result that motivated the run.
  This experiment measures the difference between two unperturbed models, not
  the growth of an injected perturbation, and five days is far short of the
  thirty over which that ratio was measured.
- Nothing below ~2e-5 m, which is where NEMO's own two time levels sit apart.
