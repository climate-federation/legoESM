# PRE-REGISTRATION — the VERDICT RUN: one year, both models, four members each

Committed BEFORE any member of this ensemble ran.  Nothing below is edited
afterwards; the result goes in a separate commit and git history proves the
order.  Probe: `verdict360.py` (self-check passes at this commit).

## The question

Every fidelity number this campaign has published is a 90-day number.  At 90
days a 1e-14 temperature kick has not saturated — the measured single-run
spread of the ACC is ~5e-06 Sv on legoESM and ~1e-05 Sv on NEMO — so *every*
gap is thousands of floors wide and "indistinguishable" is unreachable by
construction.  That is a statement about the deterministic-predictability
window, not about climate fidelity.

The question this run asks is the campaign's actual goal, the FESOM bar:

> **At a climate-relevant horizon, is the fully-fixed shipped DINO card
> statistically indistinguishable from NEMO — i.e. is the legoESM−NEMO gap no
> larger than the irreducible run-to-run spread of the two models themselves?**

## The instrument — eight runs, and what each one answers

All eight start from the SAME NEMO restart, `DINO_00005760_restart.nc`
(day 180 of the DINO spin-up), and integrate 360 days at 2700 s (11 520 steps;
NEMO kt 5761→17280).

| run | what it answers |
|---|---|
| legoESM m0 (unperturbed) | legoESM's year-360 climate on the shipped card |
| NEMO m0 (unperturbed, certified binary) | NEMO's year-360 climate from the same restart — **the twin partner; m0−m0 IS the gap** |
| legoESM m1/m2/m3 (1e-14 T kick, seeds 1/2/3) | how far legoESM moves under a perturbation far below any physical signal → legoESM's own single-run floor at each horizon |
| NEMO m1/m2/m3 (same seeds, same kick, `perturb_nemo_tn_90d.py`) | the same for NEMO → NEMO's own single-run floor |

Neither side's floor alone can judge a two-model gap: a gap is a *difference of
two runs*, so the floor it is judged against is the RSS of the two measured
spreads, `sqrt(lego_std² + NEMO_std²)`.  When the two sides wobble equally that
reduces to `sqrt(2) ×` one side's spread, which is exactly the difference
factor the 90-day lane established; the RSS is that rule without the
equal-wobble assumption.

**VERDICT RULE, fixed here and not adjustable afterwards:**
`INDISTINGUISHABLE ⇔ |gap| ≤ 2 × floor`, per metric, per horizon.

`n = 4` gives a ~41 % relative standard error on a standard deviation, so every
floor is a factor-of-two estimate and is quoted as such.  Both spread
statistics (sample std, primary; max-pairwise range, printed beside it) are
reported at every horizon; at n=4 the range runs ~2× the std BY CONSTRUCTION
and the two must never be compared across.

## Configuration — one variable against the recorded lanes: the length

**legoESM**: the shipped `nemo_dino_kamm_mlf` card at branch HEAD, **no option
flags**, launched exactly as `floor90_ensemble.run_member` launches its own
members (`run_fp64.py` wrapper, `--bridge-before`, `--save-3d`, `JAX_ENABLE_X64=1`)
with two differences and no others: `--days 360` instead of 90, and
`--snap-days 0,10,…,360` instead of the recorded `0,30,60,90` snapshot grid.
The harness's own gates run untouched: day-0 bit-identity to the NEMO restart,
the before-level bridge verify, the clean-tracked-tree provenance stamp, the
fp64 materialized-state check, NEMO's own vertical ladders (`LEGOESM_NEMO_E3T=both`),
and NEMO's absolute seasonal clock read from the restart's `adatrj`.

**NEMO**: `RUN_90D_TWIN`'s `namelist_cfg` copied byte-for-byte with exactly one
line changed — `nn_itend` 8640 → 17280 — into four fresh directories
`RUN_VERDICT360_M{0..3}`.  `namelist_ref`, `nn_it000` (5761), `nn_stock` (320,
i.e. a restart dump every 10 days), the source restart and the certified binary
`nemo.exe.certified_d3cf9242` are all copied/symlinked, never retyped.  Nothing
existing is overwritten.  16 MPI ranks each, as the 90-day twin, all four
members concurrently (64 of this machine's 80 cores).

**Snapshot grid**: 10-day on both sides, so every scored day is a day BOTH
models actually wrote state on.  No interpolation anywhere.

## Scoring — the five gate metrics, and the transports SEPARATELY

Ten metrics, every reduction imported from the recorded harness, none
re-derived:

* the five acceptance-gate metrics (`acceptance_gate_90d.metrics`): ACC,
  upper/deep meridional density contrast, southern-band surface σ max and mean;
* the channel band under BOTH recorded reductions
  (`floor90_ensemble.band_transport` = e3t_1d + median,
  `.band_transport_campaign` = e3t_0 + mean);
* the three latitude-group transports `south of band` / `band` / `north of band`
  (`acc_driver_decomp.group_transport` with its own `LAT_GROUPS` and `_avg`).

**The full-section ACC is reported but never given a per-group verdict.**  The
three latitude groups carry opposite-signed gaps — the campaign's own
decomposition has the southern basin at −0.44 Sv while the channel is positive
— so one full-section number can be two large cancelling errors.  It is
starred in the table for exactly that reason.

Two horizonal statistics, both reported, neither replacing the other:

* **endpoint**, at days 90 / 180 / 270 / 360;
* **final-90-day window mean**, days 280…360 sampled every 10 days (9 samples,
  the SAME days on both sides).  A window mean is the more climate-like
  statistic; its floor is the spread of the members' OWN window means, not the
  endpoint floor reused.

## PREDICTIONS — registered now, scored later

**P1 — the floors grow by orders of magnitude and approach saturation.**
Measured 90-day single-run ACC spreads (std): legoESM 5.40e-06 Sv, NEMO
1.02e-05 Sv (`floor90_nemo_ensemble.py`, this branch).  The 10-year
NEMO-perturbing-NEMO figure this campaign has been transferring is 0.050 Sv
(std) / 0.091 Sv (range).
*CONFIRMS* if the day-360 two-sided ACC floor lands within a factor ~3 of the
0.05–0.09 Sv class — which would also be the first validation of that
transferred constant at a horizon between 90 days and 10 years.
*REFUTES* if it is still below 1e-03 Sv at day 360 (no saturation within a
year: the ensemble is still in the deterministic window and the verdict must be
re-asked at a longer horizon) or above ~1 Sv (saturation far beyond the 10-year
figure, which would mean the 10-year constant is itself too small).

**P2 — the channel band is INDISTINGUISHABLE at 360 days.**  The channel-band
gap was +0.06 Sv at day 90, before the three bottom-drag fixes on this branch
nudged it by sub-floor amounts.  If P1 holds, the day-360 floor is ~0.05–0.09
Sv class and a channel gap of that order passes the 2× rule.
*CONFIRMS* if `|gap| ≤ 2 × floor` at day 360 on both channel reductions.
*REFUTES* if either reduction exceeds it.
Honest caveat, registered rather than discovered afterwards: this prediction is
mostly a prediction about P1.  If the floor saturates smaller than 0.05 Sv the
channel can fail on an unchanged gap, and that would be a real failure, not a
technicality.

**P3 — the southern basin: NO CONFIDENT PREDICTION, and that is the
registered position.**  The southern-basin transport gap was −0.44 Sv at day 90
on the faithful arm, and it is an equilibration feedback, not a fixed offset:
it may grow (the deficit compounds), shrink (the trajectory equilibrates toward
NEMO's), or saturate (both models reach the same attractor and the gap becomes
ensemble noise).  Every one of the three is physically reasonable and this
campaign has no measurement that discriminates them.  Registering a guess here
would be a coin flip dressed as a hypothesis.
What IS registered: the southern basin is the metric most likely to return
`no`, and its day-360 |gap|/floor ratio is the single number this run exists to
produce.

**P4 — the density metrics.**  No directional prediction.  At day 90 their gaps
sat 1–3× their 10-year transferred floors; whether a year of trajectory
divergence moves them toward or away from NEMO is exactly what is being asked.

**P5 — the full-section ACC gap at day 90 reproduces 0.382 Sv.**  legoESM
day-90 ACC at this branch HEAD was 64.986934 Sv against NEMO's recorded
65.369204 Sv (commit `59e6070a4`).  Member 0 of this run passes through day 90
on the same card, so day 90 is a free reproduction control.
*CONFIRMS* within 1e-03 Sv.  *REFUTES* otherwise — and a refutation is a
provenance failure to be chased before any day-360 number is read, not a
physics result.

## What counts as a finding rather than a failure

* A legoESM member that blows up before day 360 is a **finding**: the state at
  the failing day is printed and reported, the ensemble is NOT re-run with a
  shorter horizon to make the table complete.
* A NEMO member that stops early is the same.
* A metric whose ensemble members are all IDENTICAL has a floor of exactly zero
  by construction — the probe refuses rather than dividing by it.
* A non-finite metric anywhere is fatal.

## Cost

legoESM 90-day twin measured at ~3.7 min/member on one V100S → ~15 min at 360
days; four members, two at a time across the two GPUs, ≈ 30 min.  NEMO 90-day
twin measured at ~21 min on 16 ranks → ~85 min at 360 days; four members
concurrently on 64 of 80 cores, ≈ 2–3 h with contention.  Total ≈ 3 h, well
inside the 12 h budget, so the members run in parallel rather than serially and
no test suite touches the GPUs while the integrations hold them.

## Review

The extended harness argument (`--snap-days`) and this probe get the mandatory
dual adversarial review before any number here is cited.  The runs may start
while the review runs — the instruments decide what we believe, so no number is
final until the review clears.
