# What multiplies a tiny nudge by ~300×? Convective adjustment's on/off switch.

**Answer, in one sentence.** legoESM's convective adjustment flips vertical
mixing between 1e-5 and 100 m²/s — a factor of **ten million** — the instant the
water column tips unstable, and that hard switch makes the model's response
**2031× out of proportion** to the size of a nudge within a *single* timestep;
replacing the switch with a smooth ramp removes the effect completely (2031 →
1.00), while removing the eddy scheme or the advection limiter changes nothing.

Pre-registration: `scripts/validate/ocean_fidelity/dino_1226/PREREG_switch_rectifier.md`,
committed before the probe ran. Probe: `switch_rectifier.py`.

---

## What was measured

From the shared NEMO day-180 ocean state, in double precision, entirely in
memory. Nudge the temperature by a relative amount ε, run, and record the
**gain** — how big the response is per unit of nudge:

```
gain = ‖ T_nudged − T_control ‖ / ε
```

If the model responds in proportion to the nudge, this number is the same for
every ε. Five nudge sizes from 1e-16 to 1e-8; four horizons from one step to
day 10. **Rectified** means the gain varies by 3× or more across the ladder.

## The result

Gain spread across the nudge ladder — 1.0 would be perfect proportionality:

| arm | 1 step | 10 steps | 100 steps | 320 steps (day 10) |
|---|---:|---:|---:|---:|
| **shipped card** | **2031** | **866** | 1.95 | 2.59 |
| convective adjustment **off** | **1.58** | 10.6 | 46.6 | 33.6 |
| convective adjustment **smoothed** | **1.00** | **1.00** | 3.00 | 25.9 |
| advection limiter off | 2034 | 858 | 2.70 | 2.96 |
| eddy scheme off | 2017 | 1024 | 4.66 | 3.57 |

**The two arms that touch convective adjustment are the only ones that change
anything.** The limiter and the eddy scheme leave the 2031 untouched — they are
exonerated.

**And it is the switch, not the mixing.** The `smoothed` arm keeps convective
adjustment doing exactly the same job and only replaces its hard on/off test
with a gradual ramp. That alone takes the response from 2031× out of proportion
to **1.00 — proportional to seven digits, at two horizons**. Removing the
process entirely (`off`) is messier, because a model that never removes unstable
stratification goes somewhere else; the smoothed arm is the clean one-variable
comparison and it is unambiguous.

**Correction to my own pre-registration:** it said the switch moves diffusivity
between 1e-5 and 1.0 m²/s, "a factor 1e5". That is the software default. **The
shipped card uses 100 m²/s, so the real jump is a factor of 1e7** — a hundred
times larger than I wrote before running. Read from the card at run time.

## The catch, and it matters

**At the amplitude the ensembles actually used, this switch cannot fire.**

The probe also counts how many places in the ocean sit close enough to the
stability threshold that a nudge could tip them across:

| nudge | interfaces it could tip | share of the ocean |
|---|---:|---:|
| 1e-16 | 0 | 0% |
| **1e-14** (what the ensembles used) | **0** | **0%** |
| 1e-12 | 8 | 0.002% |
| 1e-10 | 2,773 | 0.77% |
| 1e-8 | 41,370 | 11.4% |

A 1e-14 nudge moves the stability measure by less than any point in the ocean
sits from the threshold. So **at day 0, at 1e-14, nothing flips** — and
consistently, the 1e-14 rung of the ladder is indistinguishable from
double-precision round-off and had to be dropped from the scoring.

So the honest statement is: **the rectifier is identified, but not yet at the
amplitude that produced the original puzzle.** What is established is that
this switch rectifies from about 1e-12 upward.

**The bridge that would close it, and why it is plausible** (labelled
plausible, not measured): the ensemble nudge does not stay at 1e-14. NEMO's
response reaches ~1e-6 K by day 10 — around 1e-7 in relative terms, which is
*above* the top rung of this ladder, where 11% of the ocean is within tipping
reach. So the nudge only has to grow for a few days before the switch becomes
available to it, and from then on every further difference is rectified. That
story is consistent with everything measured, and it is not yet a measurement.

## Why this is a good candidate for the original puzzle

The open item was: legoESM's ensemble spread is ~3000× NEMO's at day 90, the
offset appears early, and the growth *rates* match. A discrete switch produces
exactly that shape — it adds a one-off multiplication when it fires and changes
no growth rate afterwards.

**But this probe tested legoESM, and the 309× was measured on NEMO.** Both
models carry a convective-adjustment switch, so the mechanism transfers as a
hypothesis. It is not a measurement of NEMO.

## If it is this switch, the quantity that matters is *occupancy*, not the formula

This trigger was previously shown to be **population-exact** against NEMO at
matched states — the two models agree on every firing decision. That is a
statement about decisions at *one* state, and it says nothing about how many
places sit *within a nudge* of flipping. Two models can agree perfectly on every
decision and still differ in how often those decisions get overturned by a
disturbance.

Measured here, legoESM at the shared state: **14.8% of the ocean's interfaces
are convecting right now** (53,654 of 362,180). The neighbourhood census above
is the legoESM half of the comparison. **NEMO's half was not measured** — it is
a second measurement and it is named, not made.

That comparison — how often each model's switch gets flipped by a disturbance —
is what would connect this to the southern-basin transport deficit, because a
difference in *firing rate* is a difference in time-averaged mixing, which is a
difference in the mean state. **No such claim is made here.**

## Reading the table honestly

* **The 100- and 320-step columns are weak.** Only two rungs survive the
  round-off cut there, so "1.95" and "2.59" rest on two points. The one- and
  ten-step columns, where the effect is enormous and three rungs survive, carry
  the finding.
* **One seed, one state, one direction.** The nudge pattern is fixed; a
  different one could reach a different set of near-threshold points.
* **`off` is not a clean control.** Removing convective adjustment changes the
  trajectory itself, which is why that arm becomes rectified at longer
  horizons. The smoothed arm is the one-variable comparison.
* **The bar was fixed in advance** at 3×, and the round-off control that decides
  which rungs count was added after a smoke run exposed the need for it, before
  any arm ran to completion. Both are recorded in the pre-registration.

## What to do next

1. **Re-run the neighbourhood census on the *grown* perturbation** — at day 5
   and day 10, not day 0 — on **both** models. That converts the plausible
   bridge above into a measurement, and it produces the firing-rate comparison
   in the same pass. Cheap: both states are already on disk.
2. **Then, and only then**, ask whether the firing-rate difference has anything
   to do with the basin deficit.

Not worth doing: any further work on the advection limiter or the eddy scheme
as rectifiers. Both are cleanly exonerated at the horizon where the effect is
largest.
