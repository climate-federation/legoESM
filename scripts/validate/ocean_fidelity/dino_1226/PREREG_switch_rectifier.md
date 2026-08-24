# Pre-registration: which discrete switch multiplies a 1e-14 nudge by ~300×?

Committed **before the probe was run**. Follow-up #1 from
`dino_kick_asymmetry_result.md`. Probe: `switch_rectifier.py`.

## The observation

At day 10, NEMO's response to a 1e-14 relative temperature nudge is **not
proportional to the nudge**. Six perturbation sizes span a factor of **309**;
only two distinct locations carry the largest perturbation across three
members; and seeds 1 and 3 — different restarts, genuinely different fields —
agree in perturbation size to **5 parts in a million at the same cell**.
Independent draws cannot do that. Something discrete is setting the amplitude.

A discrete switch that fires or does not depending on the nudge would produce
exactly the recorded signature of the original open item: an offset that
appears immediately and leaves the subsequent growth *rates* untouched.

## The test

From the shared NEMO day-180 state, legoESM, in double precision, in memory
(no single-precision snapshot anywhere in the path). For each nudge size
ε in **1e-14, 1e-12, 1e-10** — the ladder that spans the observed 309× — run
the model and record

```
G(ε, n) = || T_perturbed(n steps) − T_control(n steps) ||_2  /  ε
```

at n = **1, 10, 100, 320** steps (320 = day 10, where the 309× was observed).
`G` is the **gain**: with a response proportional to the nudge it is the same
number for every ε.

**Linearity statistic**, per arm and per step count:

```
spread = max(G over the three ε) / min(G over the three ε)
```

## The pre-registered rule

* **spread < 3** — proportional. No rectification at that horizon.
* **spread >= 3** — rectified. Something discrete is in the path.

3 is chosen because it is comfortably below the observed 309 and comfortably
above what a well-conditioned linear operator produces (1.0 up to round-off).
`G` itself is reported next to the spread so a reader can see whether an arm
changed the gain as well as its ε-dependence.

**THE RECTIFIER is the arm whose disabling collapses the spread below 3 at the
step count where the baseline exceeds it.** If no arm does, the rectifier is
not in the list and the finding is that list's incompleteness.

## The arms — one process each, everything else the shipped card

| arm | what changes | why |
|---|---|---|
| `baseline` | nothing | the shipped card |
| `evd_off` | convective adjustment removed | the prime suspect |
| `evd_smooth` | convective adjustment kept, its hard threshold replaced by a smooth ramp | separates the **process** from its **discontinuity** — if this collapses the spread but `evd_off` also does, the discontinuity is the rectifier, not the mixing |
| `limiter_off` | tracer advection's flux limiter removed, same underlying high-order flux | the limiter's min/max branches |
| `gm_redi_off` | eddy parameterisation removed | its slope cap and mixed-layer-depth index |

Found by grepping the tracer-to-density path for threshold and branch
processes active in this card. **The lever arm on the prime suspect is
enormous and is stated here in advance:** convective adjustment switches
vertical diffusivity between 1e-5 and 1.0 m²/s — a factor **1e5** — at a hard
`N² < −1e-12` test, with the smooth-transition option **off** in this card.
One interface crossing that threshold is more than enough to make a 1e-14
nudge visible.

## What this probe does NOT settle

* **It tests legoESM.** The 309× was measured on **NEMO**. Both models carry a
  convective-adjustment switch, so the mechanism transfers as a hypothesis, not
  as a result. A legoESM rectifier at the same amplitude is strong
  circumstantial evidence; it is not a measurement of NEMO.
* **A null is weak.** Rectification could need many steps, or a different
  perturbation direction, or could live in a process not on the list.
* **If the switch is convective adjustment, the finding is about its
  THRESHOLD NEIGHBOURHOOD, not its formula.** The trigger was previously shown
  to be population-exact against NEMO at matched states. Exactness at a state
  says nothing about behaviour under perturbation: two models can agree on
  every firing decision at one state and still differ in how many cells sit
  *within a nudge* of flipping. That occupancy — how many interfaces have |N²|
  below the amount the perturbation moves N² by, in each model — is the
  rectification-rate comparison that would connect this to the basin deficit,
  because a difference in firing *rate* is a difference in time-mean mixing.
  It is reported here as a census on both models at the shared state; it is
  **not** a claim about the deficit.

---

## AMENDMENT, after a harness smoke test and before any arm ran to completion

A 10-step smoke run of the `baseline` arm — done to check the harness threads
the card's surface forcing correctly, which it did not first time, and the
model's own guard raised — produced gains that exposed a **missing control**.
The ladder is changed before the real run.

The smoke numbers, recorded because they are measurements and not to be quietly
re-taken:

| nudge | G(1 step) | G(10 steps) |
|---|---:|---:|
| 1e-14 | 1.6859e+04 | 1.4449e+05 |
| 1e-12 | 5.8441e+03 | 5.8285e+03 |
| 1e-10 | 1.1868e+07 | 5.0449e+06 |

Two things are visible; only one is trustworthy.

* **1e-10 is 2031× off 1e-12 at a single step.** No round-off produces that.
  Rectification within one step is real at that amplitude.
* **1e-14 sits 2.9× *above* 1e-12** — the wrong direction for a nonlinearity
  that grows with amplitude, and exactly what an **arithmetic floor** looks
  like. A 1e-14 relative nudge is ~2e-13 K on a ~20 K field against ~4e-15 K of
  double-precision round-off per cell: only ~45× clear. Not enough to quote.

**The fix, registered before the run.** Five rungs, **1e-16 … 1e-8**, with
**1e-16 as a round-off control**: at that size the kick is at round-off, so the
response it produces *is* the arithmetic floor. A rung whose **response**
`G × ε` is within **3×** of the control's is measuring that floor and is
excluded from the spread. The number of rungs actually used is printed in every
column, so an arm scored on two rungs cannot be mistaken for one scored on four.

*Note on the comparison, because getting it backwards is easy and the first
implementation did:* the floor is a floor on the **response**, not on the
**gain**. At `ε = 1e-16` the gain is enormous precisely because ε sits in the
denominator, so comparing *gains* would exclude every honest rung and keep the
floor — the exact inversion. The self-check asserts the exclusion flips the
verdict **both ways**: a floor-contaminated ladder reads rectified without it
and proportional with it, and a genuinely rectified ladder survives it.

A control was added; the bar was not moved. `spread >= 3` still decides, now
computed only over rungs that clear the floor.

**Consequence for the reading.** The question sharpens: does rectification
appear at the **ensemble's own 1e-14 amplitude**, or only far above it? Only the
first would explain the recorded 309×. Three rungs could not tell; five can.

---

## AMENDMENT 3 (2026-08-24) — POST-HOC statistic declared, and the registered rule's own answer

Written **after** the six arms were scored. Everything here is post-hoc and is
labelled as such.

### 3a. The registered rule, applied literally, isolates NOTHING

The rule was: *the rectifier is the arm whose disabling collapses the spread
below 3 at the step count where the baseline exceeds it.* The baseline exceeds
3 at **all four** horizons (2031 / 866 / 1540 / 1869). Measured:

* `evd_off` collapses at n=1 only (1.58), then 10.6 / 46.6 / 33.6.
* `evd_sharp` collapses at n=10 only (1.83), with 524 / 52.0 / 369 elsewhere.

**No arm collapses at every horizon where the baseline is rectified, so under
the registered rule no arm is identified as the rectifier.** That is the
pre-registered answer and it is stated first in the result, ahead of anything
post-hoc.

### 3b. The adjacent-rung pair ratio is POST-HOC

`gain(1e-10) / gain(1e-12)` was chosen **after seeing the ladder**, because the
full-ladder spread mixes two effects. It is a legitimate statistic and it is
not a registered one. It is reported as post-hoc, never as the registered
verdict.

Its originally stated justification — that both rungs are "small enough not to
outrun the narrow ramp" — is **wrong, by arithmetic**: a 1e-10 nudge moves N²
by 3.673e-12, which is **37× the 1e-13 ramp width**. Both rungs outrun the
ramp; only the degree differs (1e-8 outruns it by 3700×). The pair is therefore
an *empirical observation about two adjacent rungs*, not an a-priori-safe
bracket.

### 3c. The `evd_sharp` arm was not one-variable either — and the fix

The smoothing branch is `sigmoid(-N2 * sharpness)`: centred on **N² = 0**, and
it **ignores `n2_threshold`**, while the hard branch fires at `N² < −1e-12`.
So `evd_sharp` differs from the shipped card in edge sharpness **and** trigger
location, and the census shows 0.577% of wet interfaces piled into exactly the
band they disagree about — each able to move by up to 1e7×. The claim "mixing
field unchanged to the digit" was false for the very population carrying the
signal.

**Fix, registered here before it was scored:** the arm `evd_thr0` is the
shipped **hard** switch relocated to `N² < 0`. `evd_sharp` vs `evd_thr0` then
differ **only** in edge sharpness, and that pair — not `evd_sharp` vs baseline
— is the one-variable discontinuity comparison.

### 3d. Honest scope, tightened

At 1e-14 zero interfaces are within tipping reach and at 1e-12 only **eight**,
and "reach" is a global maximum compared against a per-interface distance, so
eight is an **upper bound**. The first rung with a substantial population is
1e-10. The defensible scope is therefore **"a rectifier at ≳1e-10 in legoESM,
silent at the ensembles' 1e-14"** — which means it does **not**, on this
evidence, explain NEMO's recorded 309×.

### 3e. Known non-blocking limitations, recorded not fixed

* **Rung sets differ across arms** (`evd_off` keeps 4, the others 3), so
  cross-arm full-ladder spreads are not strictly matched. The adjacent-rung
  pair is matched, because those two rungs survive in every arm.
* **Arm validity is measured at t=0 only.** `evd_sharp`'s gains at n=10 run
  1e6–1e9 against the baseline's 5e3–1e4, i.e. its trajectory diverges; the
  fix is to print the distance between the two arms' *unperturbed* controls at
  each horizon, which needs no new run, only storing them.
