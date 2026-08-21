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

---

# ERRATUM — written after DUAL ADVERSARIAL REVIEW, BEFORE any member finished

Committed separately.  Everything above is unedited; git history proves that
this text was written while the members were still integrating and before a
single metric had been scored.  Both reviewers read the pre-registration and
the runner at `4e652b101`.  Five things they found are corrected here rather
than by editing the pre-registered text.

## E1 — a dead formula in the flip-point paragraph

The sentence beginning "if the measured floor is **smaller** than 0.091/5 x
(0.455 - 0.044) / 5 ..." contains a garbled fragment that evaluates to
~0.0015 and means nothing.  The **correct and pre-registered flip point is the
one the very next sentence derives: 0.4107 / 5 = 0.0821 Sv.**  The fragment is
struck, not the conclusion.

## E2 — the floor measured here is a LOWER BOUND, and the decision table's "hardens" branch is the one that suffers

The perturbation touches the now-level temperature and nothing else — not
salinity, not the velocities, not the free surface, and not the before-level
fields.  That samples ONE direction of a many-dimensional tangent space, and
the kick needs time to spread into the fields the transport metrics actually
integrate.  Both effects push the same way:

> **the spread measured here UNDER-estimates the true single-run floor.**

The pre-registration disclosed the T-only restriction but never named the
direction of the bias, which is the part that matters, because an
under-estimated floor makes every `gap / floor` ratio look BIGGER than it is —
it flatters exactly the "statistically distinguishable" claims this lane
exists to referee.

Consequence for the pre-registered decision table, recorded before the numbers
exist: the `> 0.2 Sv` branch (verdict WEAKENS) is **robust** to this bias — a
lower bound above 0.2 Sv means the true floor is at least that.  The
`< 0.05 Sv` branch (verdict HARDENS) is **not** robust: a small measured
spread is exactly what an under-sampled perturbation subspace produces, so
that branch is downgraded in advance from "the verdict hardens" to "the
verdict is not contradicted, and the floor is not resolved from below."

## E3 — max-pairwise IS the sample range, and runs ~2x the std BY CONSTRUCTION

The pre-registration says both spread statistics are reported and "neither is
privileged", which implies they are two views of one number.  They are not.
Over n samples the max over all pairs is identically `max - min`, the sample
RANGE, and for normal samples at n=4 the expected range is ~2.06 sigma.  So
max-pairwise is expected to be about twice the std for no physical reason
whatsoever.  Both are still reported, but the ~2x is structural and must not
be read as one statistic "capturing more".

## E4 — the 10-year constants are RANGE statistics measured on NEMO, not on legoESM

Traced to `docs/ocean/fidelity/dino_1226_state.md` (the #1492 item-2.1 record).
Two facts there change how the comparison must be written, and neither was in
the pre-registration:

1. **The gate's constants are ranges, not standard deviations** — specifically
   the MAX over branch-years 1-10 of a 3-member range.  The same table records
   the corresponding std maxima: ACC 0.050 Sv, upper contrast 5.7e-5, deep
   contrast 2.3e-5, southern surface sigma max 5.3e-5.  So the apples-to-apples
   comparator for the gate's 0.091 Sv is this lane's **max-pairwise**, and the
   std column must be compared against the recorded std column, never against
   0.091.  Both comparisons are now printed.  (The 10-year number is still a
   MAX OVER TEN YEARLY SAMPLES of a range, while this lane measures a single
   range at day 90, so even the matched comparison is biased in the 10-year
   number's favour.)
2. **That ensemble was NEMO perturbing NEMO.**  The constants describe NEMO's
   internal variability on a 10-year branch.  This lane measures legoESM's
   internal variability on a 90-day branch.  They are different quantities
   about different models at different horizons, and the gate has been applying
   one as if it were the other.

E4(2) also repairs the `sqrt(2)` rule, which the physics review correctly
flagged as resting on an unstated equal-variance assumption.  The honest
combination for a legoESM-minus-NEMO gap is

    difference floor = sqrt( floor_lego^2 + floor_nemo^2 )

and both terms are now measured quantities rather than one measured and one
assumed equal to it.  `sqrt(2) x floor_lego` is the special case where the two
are equal.  Because the NEMO term is only available at the 10-year horizon,
the combination remains a cross-horizon transfer and is labelled as one
everywhere it is used.

## E5 — two mechanical fixes to the runner

* The land mask is now asserted IDENTICAL across all four members before any
  metric is compared.  A silently diverging wet-cell set would not have raised;
  it would have compared members on different domains.
* The channel-band metric is reported under TWO reductions, because the
  pre-registered one is not the one the campaign has been quoting.  The
  campaign's `+0.2874 Sv` arm-A band gap is reproduced EXACTLY by e3t_0
  weighting with the MEAN over longitudes 2..-2; the pre-registered
  e3t_1d + median reduction gives `+0.2791 Sv` on the same artifact.  Both are
  legitimate; they are different reductions of the same state, not a
  discrepancy, and quoting a floor under one while the campaign quotes gaps
  under the other would be a protocol mismatch of exactly the kind this lane
  is auditing.

## E6 — the decision table's bands are finer than the instrument

At n=4 the floor is a factor-of-two estimate, and the middle band spans only a
factor of four.  Any measured floor within a factor of 2 of a band boundary is
reported as STRADDLING two bands, and the weaker of the two consequences is the
one adopted.

---

# RESULT

Four members at `4e652b101`, clean tracked tree, shipped card
`nemo_dino_kamm_mlf` with NO option flags (the card's own defaults resolve to
`barotropic_reconcile_target="velocity_avg"` + `barotropic_after_reconcile=
"nemo_mlf_baro_corr"`, the faithful pair), `--bridge-before --save-3d`, IC =
NEMO `DINO_00005760_restart.nc` (day 180), twin-default ladder `both`,
corrected clock `t0 = 15552000 s`, fp64 control dtype, dt 2700 s, 90 days =
2880 steps, `CUDA_VISIBLE_DEVICES=0`, ~211 s each.  **All four STABLE to day
90**; each log carries the three stamps and its own perturbation line
(`max_rel|dT/T|` = 5.04e-14, 5.18e-14, 4.58e-14).

## Instrument controls — all four passed

1. Gate self-checks: NEMO y10 ACC 121.07 Sv; band volume 2.694775e16 m3
   (rel 4.98e-08).
2. **The unperturbed control reproduces the recorded shipped-card ACC gap:
   0.410731 Sv against the recorded 0.4107, |delta| 3.1e-05 Sv.**
3. Every member printed its own `DONE nsteps=2880 STABLE=True`, all four at one
   SHA, land mask identical across all four.
4. Every metric separates at least two members, and the growth control below
   shows the kick reached the integrator.

**The growth control — the measurement that decides how to read everything
below.**  Peak temperature difference of each perturbed member from the
control, by snapshot day (float32 snapshots: the ~2e-06 K quantum is why day 0
reads zero; the real day-0 kick is ~1e-12 K):

| member | day 0 | day 30 | day 60 | day 90 |
|---|---|---|---|---|
| seed 1 | 0 | 1.91e-06 | 7.44e-05 | 7.94e-04 |
| seed 2 | 0 | 2.86e-06 | 3.69e-04 | 9.77e-03 |
| seed 3 | 0 | 2.86e-06 | 3.69e-04 | 9.51e-03 |

The perturbation grows by roughly one order of magnitude per 30 days — real,
active error growth, so the tiny spread below is NOT the artifact of a kick
that never propagated.  But at day 90 the divergence is still ~1e-02 K on a
26 K field, i.e. **~3 parts in 10,000 — nowhere near saturation.**  The 10-year
ensemble the gate's constants come from saturated in years 3-5.

## THE FLOOR TABLE

Single-run 90-day floor, four members.  The gate's constants are RANGE
statistics measured on NEMO over ten years; the range column is therefore the
apples-to-apples comparator and the std column is compared against the
recorded 10-year std.

| metric | 10-yr floor IN USE (range) | 10-yr std | **90d max-pairwise** | **90d std** | ratio (range) | ratio (std) |
|---|---|---|---|---|---|---|
| ACC [Sv] | 0.091 | 0.050 | **1.146e-05** | **5.40e-06** | 1.26e-04 | 1.08e-04 |
| upper contrast [kg/m3] | 1.1e-04 | 5.7e-05 | **5.00e-08** | **2.56e-08** | 4.54e-04 | 4.50e-04 |
| deep contrast [kg/m3] | 4.5e-05 | 2.3e-05 | **2.90e-09** | **1.33e-09** | 6.45e-05 | 5.78e-05 |
| S-band sigma MAX [kg/m3] | 9.5e-05 | 5.3e-05 | **1.52e-07** | **7.60e-08** | 1.60e-03 | 1.44e-03 |
| S-band sigma MEAN [kg/m3] | 9.5e-05 | 5.3e-05 | **1.53e-07** | **7.46e-08** | 1.61e-03 | 1.41e-03 |
| channel band, e3t_1d median [Sv] | none | none | **2.18e-05** | **1.02e-05** | — | — |
| channel band, e3t_0 mean [Sv] | none | none | **1.78e-05** | **7.69e-06** | — | — |

**The floor in use is between 600x and 8,000x too large for the horizon it is
applied at.**  On ACC specifically the constant is 0.091 Sv and the measured
90-day spread is 1.15e-05 Sv — a factor of **7,900**.

Two bounds, both stated because they point in opposite directions and neither
changes the conclusion:

* the T-only perturbation samples one direction of the tangent space, so this
  is a **lower** bound on the true single-run floor (E2);
* the 3-D snapshots are stored float32 and the accumulated rounding of the
  stored velocity field is the same order as the measured ACC spread, so as
  measured through this artifact it is also an **upper** bound (the probe
  prints the quantum next to the number).  Two of the four members tie exactly
  on the sigma-MAX metric, which is that quantum showing itself.

Neither bound is worth a factor of 7,900.

## THE RE-SCORED VERDICTS

Difference floor on ACC = `sqrt(2) x 1.146e-05` = **1.62e-05 Sv**
(against `sqrt(2) x 0.091` = 0.129 Sv on the transferred constant).

### The five gate metrics + the channel band, shipped card

| metric | gap vs NEMO | ÷ (sqrt2 x 10-yr, TRANSFERRED) | ÷ (sqrt2 x 90-day, MEASURED) | 5x gate on the 90-day floor |
|---|---|---|---|---|
| ACC [Sv] | 0.41073 | 3.19 | **2.53e+04** | FAIL by 7,200x |
| upper contrast | 2.367e-04 | at floor (1.52) | **3.35e+03** | FAIL |
| deep contrast | 1.544e-06 | at floor (0.02) | **3.76e+02** | FAIL |
| S-band sigma MAX | 1.072e-04 | at floor (0.80) | **4.99e+02** | FAIL |
| S-band sigma MEAN | 2.943e-04 | 2.19 | **1.36e+03** | FAIL |
| channel band (e3t_1d, med) | 0.0669 | n/a | **2.17e+03** | FAIL |
| channel band (e3t_0, mean) | 0.0632 | n/a | **2.51e+03** | FAIL |

The pre-registered flip point was 0.0821 Sv: any measured floor below it means
the shipped card does not pass its own ACC gate once that gate is calibrated at
90 days.  The measured floor is 1.15e-05 Sv, **four orders below the flip
point**, and every one of the six metrics fails a 90-day-calibrated 5x gate.

**This does NOT mean the shipped card is bad.**  It means the gate's threshold
has never been a 90-day noise floor and cannot be read as one — see the
recommendation.

### RETRACTIONS — three round-2 corrections were right arithmetic on the wrong floor

`PHASE2_R6_alignment_and_prereg.md`'s round-2 section corrected three
over-claims by re-scoring them against `sqrt(2) x 0.091 = 0.129 Sv`.  That
correction was itself the third floor error it was written to fix: it
transferred a 10-year floor to a 90-day comparison, which the same section
explicitly named as unvalidated in its own point 3.  Re-scored against the
measured 90-day floor:

| claim | round-2 verdict | ÷ measured difference floor | now |
|---|---|---|---|
| B vs D on ACC, 0.107 Sv | "NOT separable" | **6.6e+03** | **RETRACTED — B and D are separable, decisively** |
| A -> C average axis, 0.027 Sv | "climate-inert by itself" | **1.7e+03** | **RETRACTED as a floor statement — the move is readable; whether it is climatically important is a different question the floor cannot answer** |
| interaction contrast, 0.1336 Sv | "right at the edge, not resolved" | **8.2e+03** | **RETRACTED — the interaction IS resolved** |
| B vs D on all four density metrics | "indistinguishable" | 190x, 1140x, 72x, **24x** | **RETRACTED — all four are readable** |

The measured second-site effects (A->B 0.292 Sv, C->D 0.159 Sv) are 1.8e+04 and
9.8e+03 difference-floors: those were already called readable and they harden.

**Robustness of the retractions to the floor's own uncertainty.**  n=4 gives a
41% relative standard error and the T-only sampling makes the number a lower
bound, so the honest test is how much the floor would have to be wrong by.
Every ACC retraction survives inflating the measured floor by **1,000x**.  The
weakest entry in the table, the B-vs-D sigma-MEAN comparison at 24
difference-floors, survives a **10x** inflation but not a 100x one; it is the
only one flagged as sensitive.

## RECOMMENDATION — a proposal, not a change

No constant in `acceptance_gate_90d.py` was edited by this lane and none should
be edited on the strength of this measurement alone.  **The 0.091 Sv constant
should not be replaced by 1.15e-05 Sv.**  Doing so would fail every
configuration this project will ever produce, including a perfect port,
because the gate compares two different codes — NEMO in Fortran against
legoESM in JAX — and the irreducible difference between two faithful
implementations of the same equations has never been measured.  The measured
run-to-run floor bounds legoESM's reproducibility, not the cross-implementation
agreement the gate is actually testing.  What is wrong is the LABEL, not the
number: the gate calls its threshold a noise floor, and a 5/5 PASS is therefore
read as "indistinguishable from NEMO", when what it means is "within five times
a tolerance borrowed from a different model at a different horizon".  The
proposal has four parts, in priority order.  (1) Rename the constant to what it
is — a declared acceptance TOLERANCE — and record its provenance in the gate's
own docstring, so no future reader infers statistical indistinguishability from
a PASS.  (2) Keep the numeric values; as a tolerance they are defensible and
changing them would silently rescore the whole campaign.  (3) Add the measured
90-day reproducibility floor as a SECOND, separate bar used only for
legoESM-versus-legoESM A/B comparisons, where it IS the correct bar and is four
orders tighter — this is the bar that should have scored the R6 square, and
using the tolerance there is what produced the three retractions above.  (4) If
a genuine 90-day noise gate is wanted, the missing measurement is NEMO's own
90-day 3-member ensemble from the same day-180 restart; it costs three NEMO
runs and would supply the `floor_nemo` term that the `sqrt(2)` rule currently
has to assume.

## Also recorded

The promotion lane's arm-D twin (`/tmp/dino_valid/q_D.npz`, produced at
`d27dc0909` with explicit env overrides) is **BIT-IDENTICAL** to the
unperturbed member on every one of the seven metrics (|diff| exactly 0.0).
The concern that motivated re-running member 0 — three intervening commits,
one of them a numerics fix to the default barotropic filter — was legitimate
and the answer is null: on this configuration those commits moved nothing.
Reaching the faithful pair through the card's defaults gives the same
trajectory as reaching it through the env overrides.
