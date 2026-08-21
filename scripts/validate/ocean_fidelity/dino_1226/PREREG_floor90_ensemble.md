# PRE-REGISTRATION — the TRUE 90-day noise floor of the acceptance-gate metrics

Written and committed BEFORE any member runs.  Nothing above the RESULT heading
is edited afterwards; git history proves the order.

---

## Why this exists

`acceptance_gate_90d.py` scores five metrics against floor constants

    ACC 0.091 Sv | upper 1.1e-4 | deep 4.5e-5 | sigma max 9.5e-5 | sigma mean 9.5e-5

taken from the #1492 2.1 micro-ensemble.  That ensemble was measured on
**ten-year** branches.  The gate applies those constants to **90-day** twins.
**The transfer across horizons was never validated.**  A floor is not a property
of a model; it is a property of a model AND a horizon AND a metric, and a
90-day floor and a 10-year floor have no reason to be equal.

Two live verdicts rest on the transfer:

* the shipped card (arm D, the faithful pair) passes the ACC metric at the 5x
  level with **0.044 Sv of margin** (gap 0.4107 Sv against a 5 x 0.091 =
  0.455 Sv threshold).  A floor 10% larger changes nothing; a floor 10%
  *smaller* fails the shipped card.
* the R6 campaign calls the 0.411 Sv arm-D ACC gap "statistically
  distinguishable" at 4.5 floors, and calls the A->C average axis
  (0.027 Sv) "climate-inert" at 0.3 floors.  Both statements are floor
  arithmetic and both are re-scored by this measurement.

## Design — one variable, four members

A 3-member perturbed ensemble plus the unperturbed control, **all four at the
same source SHA**, on the SHIPPED card with no option flags:

| member | perturbation |
|---|---|
| m0 | none (control) |
| m1 | `--perturb-seed 1` |
| m2 | `--perturb-seed 2` |
| m3 | `--perturb-seed 3` |

`kamm_twin_90d.py`'s `--perturb-seed` applies, **after** the NEMO bridge and
**before** step 0, a per-grid-point multiplicative perturbation to the
now-level temperature only:

    T <- T * (1 + 1e-14 * N(0,1)),   rng = numpy.random.default_rng(seed)

i.e. a relative perturbation at the 1e-14 level, ~50x the fp64 epsilon and
~13 orders below any physical signal.  It does NOT touch S, u, v, eta, or the
before-level fields; the runner records the harness's own printed
`max|dT|` / `max_rel|dT/T|` line for each member so the size of the kick is in
the artifact, not asserted.  This is the same eps and draw shape as the
recorded `scripts/tmp/_perturb_restart_ensemble.py` pattern.

Held identical and stamped in every member log: HEAD sha, clean tracked tree,
recipe `nemo_dino_kamm_mlf`, `--bridge-before`, `--save-3d`, IC = NEMO
`DINO_00005760_restart.nc` (day 180), the twin-default vertical ladder
(`both` = NEMO's own thickness AND T-depth ladders), the corrected absolute
seasonal clock (`t0 = 15552000 s` from the restart's own `adatrj`), fp64
control dtype via `run_fp64.py`, dt 2700 s, 32 steps/day, 90 days = 2880 steps,
`CUDA_VISIBLE_DEVICES=0`.

**Member 0 is RE-RUN rather than reusing `/tmp/dino_valid/q_D.npz`.**  That
artifact's stamps are correct (clock `15552000`, ladder `both`, control dtype
`float64`) but it was produced at `d27dc0909`, three commits behind HEAD, and
one of the three (`453857c64`) is a numerics fix to the default barotropic
filter.  An ensemble spread whose control sits at a different SHA from its
members measures the SHA difference as well as the noise.  The run costs
~3.5 minutes, so the confound is not worth carrying.  The old artifact is
scored alongside as a cross-SHA reproducibility check, reported separately and
never mixed into the spread.

## Metrics scored — five gate metrics + the channel band

The five gate metrics come from `acceptance_gate_90d.metrics`, IMPORTED, not
re-derived, so the floor is measured on exactly the quantity the gate scores:
`acc`, `up`, `deep`, `smax`, `smean`.

The sixth, the **channel-band transport**, is
`acc_thermal_wind.acc_band(u, wet_u, e3=e3t_1d)` reduced by the median over
longitudes 2..-2 — the same weighting and the same reduction `acc_full` uses,
restricted to the re-entrant band.  It is scored because the campaign quotes a
band number (`+0.2874 Sv` for arm A) next to every ACC number and that number
has never had a floor at all.

## The floor estimator, fixed in advance

For each metric, over the four members:

* **max pairwise** = max over the 6 unordered pairs of |m_i - m_j|
* **std** = `numpy.std(..., ddof=1)` over the 4 member values

BOTH are reported for every metric.  Neither is privileged.

**n=4 caveat, stated in advance and repeated at every use.**  The relative
standard error of a standard deviation estimated from n samples is
approximately `1/sqrt(2(n-1))`; at n=4 that is **41%**, and the max-pairwise
statistic is more volatile still.  Every floor number this note produces is
therefore a *factor-of-two* estimate.  It is quoted to three digits because
that is what the arithmetic produces, NOT because it is known to three digits.
No conclusion in the RESULT section may depend on a floor ratio between 0.5 and
2.0.

## FLOOR ARITHMETIC — the three rules, applied mechanically

This campaign made three floor errors in one lane, all self-flattering.  The
rules are now mechanical and every sentence in the RESULT section that uses a
floor states which one it is applying.

1. **A difference of two runs is compared against `sqrt(2) x` the single-run
   floor.**  The ensemble spread measured here is a single-run floor: it is the
   scatter of *one* legoESM twin about its own attractor.  A legoESM-minus-NEMO
   gap is a difference of two independent realisations, so the bar it must
   clear is `sqrt(2) x floor`, not `floor`.  Every "gap / floor" ratio below is
   reported as `gap / (sqrt(2) x floor)` and labelled as such.
2. **Never read a ratio off two numbers that are both below their floors.**
   If a gap and a floor are within a factor of ~1 of each other, the ratio is
   reported as "at the floor" and no multiple is quoted.
3. **Never transfer a floor across horizons without saying so.**  The 10-year
   constants are labelled "10-yr, transferred" everywhere they appear next to a
   90-day number, including in the ratio columns computed with them.

## Decision rules, written before the numbers exist

For the ACC metric, whose gap on the shipped card is **0.4107 Sv**:

| measured 90-day ACC floor | consequence for the "statistically distinguishable" verdict |
|---|---|
| **> 0.2 Sv** | gap / (sqrt(2) x floor) < 1.5 — the verdict **WEAKENS to "at the floor"**; the R6 second-site effect (0.159-0.292 Sv) becomes unreadable and its "3.2x floor" claim is retracted |
| 0.05 - 0.2 Sv | gap is 1.5-5.8 difference-floors — the verdict **STANDS but the multiple is restated**; the A->C "climate-inert" call is re-checked against the new floor |
| **< 0.05 Sv** | gap > 5.8 difference-floors — the verdict **HARDENS**; the 10-yr constant was conservative and the gate is looser than it needs to be |

For each of the other five metrics the same three-band logic applies with the
metric's own shipped gap substituted, and specifically for the **marginal 5th**
(the metric whose 5x-level margin is smallest — ACC, at 0.044 Sv):

* if the measured 90-day floor is **larger** than 0.091 Sv, the 5x threshold
  widens and the shipped card's PASS becomes *more* comfortable, but the
  *scientific* claim built on the same floor gets *weaker* — these move in
  opposite directions and both are reported;
* if the measured floor is **smaller** than 0.091/5 x (0.455 - 0.044) / 5 ...
  concretely: the shipped card FAILS the 5x gate at any floor below
  **0.4107 / 5 = 0.0821 Sv**.  That number is pre-registered as the flip point.
  A measured floor below 0.0821 Sv means **the shipped card does not pass its
  own gate once the gate is calibrated at the horizon it is applied at.**

## What is NOT done here

* **No gate constant is edited.**  `FLOORS` in `acceptance_gate_90d.py` is
  owner-controlled.  This lane produces a written proposal and nothing else.
* **No card default is edited.**
* The probe prints tables, never verdicts.  Interpretation lives in the RESULT
  section below, after the instrument's own controls have passed.

## Instrument controls that must pass before any number is quoted

1. `acceptance_gate_90d.py`'s two fatal self-checks (NEMO y10 ACC 121.07 Sv;
   band volume 2.694775e16 m3) run inside the scorer.
2. Member 0 (unperturbed, HEAD) must reproduce the recorded arm-D ACC gap of
   **0.4107 Sv** to within the measured floor.  If it does not, the three
   intervening commits moved the trajectory and THAT is the finding.
3. Every member must be stable to day 90 and carry the three stamps.  A blown
   member is a finding, not a member to be dropped.
4. Every member's perturbation line (`max_rel|dT/T|`) must be ~1e-14 and the
   three seeds must produce three DIFFERENT day-90 states.  Identical states
   would mean the perturbation never reached the integrator.

---
