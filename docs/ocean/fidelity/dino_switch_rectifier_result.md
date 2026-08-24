# What multiplies a tiny nudge by ~2000×? Convective adjustment's on/off switch.

**Answer, in one sentence.** legoESM's convective adjustment flips vertical
mixing between 1e-5 and 100 m²/s — a factor of **ten million** — the instant a
water column tips unstable, and replacing *only that hard edge* with a ramp
narrow enough to leave the mixing field otherwise identical takes the model's
response from **2031× out of proportion** to the size of a nudge down to
**1.006×**, while removing the advection limiter or the eddy scheme changes
nothing at all.

Pre-registration: `scripts/validate/ocean_fidelity/dino_1226/PREREG_switch_rectifier.md`,
committed before the probe ran, with two amendments both recorded before
scoring. Probe: `switch_rectifier.py`. Raw output: the six arms below were
produced in **one run, at one code state**.

---

## What was measured

From the shared NEMO day-180 ocean state, in double precision, entirely in
memory — no single-precision snapshot anywhere in the path. Nudge the
temperature by a relative amount ε, run, and record the **gain**, i.e. how big
the response is per unit of nudge:

```
gain = ‖ T_nudged − T_control ‖ / ε
```

If the model responds in proportion to the nudge, this is the same number for
every ε. Five nudge sizes 1e-16 … 1e-8; four horizons, one step to day 10.

## The decisive table

Gain at 1e-10 divided by gain at 1e-12, after a single timestep. **1.0 means
proportional.** These two rungs are the step where the shipped card jumps; both
clear the round-off floor at every horizon, and neither is large enough to
outrun the narrow ramp (see the caveat below).

| arm | what it changes | gain ratio |
|---|---|---:|
| **shipped card** | nothing | **2030.8** |
| advection limiter off | limiter removed, same high-order flux | 2034.4 |
| eddy scheme off | GM/Redi removed | 2017.5 |
| convection off | process removed entirely | 1.0026 |
| **convection, hard edge → narrow ramp** | **only the discontinuity** | **1.0063** |

The limiter and the eddy scheme leave the 2031 completely untouched — they are
**exonerated at every horizon**, not just this one. Convective adjustment is
the owner, and the last row says it is specifically the **edge**, not the
mixing: that arm keeps convective adjustment doing the same job at the same
strength and only softens the threshold.

**How narrow, and how do we know it is one variable?** The probe measures it
rather than asserting it. The ramp's width in stratification is 1e-13, which is
below the switch's own −1e-12 trigger offset:

| arm | interfaces inside the ramp | median mixing change | interfaces whose mixing moves >10% |
|---|---:|---:|---:|
| ramp at default width (1e-6) | 40.4% | **31,000×** | 77.3% |
| **narrow ramp (1e-13)** | **0.577%** | **1.000×** | **0.611%** |

**Retraction.** An earlier version of this document ran only the default-width
ramp and reported its collapse (2031 → 1.00) as proof that the discontinuity is
the rectifier. That arm is **invalid** — it mixes 40% of the ocean interior
31,000× harder than the shipped card, so it is a different ocean, not a
softened switch. A reviewer flagged it and the probe's own arm-validity census
now confirms it with the numbers above. The conclusion survives, but it rests
on the narrow-ramp arm; the earlier evidence did not support it.

**Correction to my own pre-registration:** it said the switch moves diffusivity
between 1e-5 and 1.0 m²/s, "a factor 1e5". That is the software default. **The
shipped card uses 100 m²/s, so the real jump is a factor of 1e7** — a hundred
times larger than I wrote before running.

## The catch, and it is load-bearing

**At the amplitude the ensembles actually used, this switch cannot fire.** The
probe counts how many places sit close enough to the stability threshold that a
nudge could tip them:

| nudge | interfaces it could tip | share of the ocean |
|---|---:|---:|
| 1e-16 | 0 | 0% |
| **1e-14** (what the ensembles used) | **0** | **0%** |
| 1e-12 | 8 | ≤0.002% |
| 1e-10 | 2,773 | ≤0.77% |
| 1e-8 | 41,370 | ≤11.4% |

A 1e-14 nudge moves the stability measure by less than any point in this ocean
sits from the threshold. Consistently, the 1e-14 rung of the ladder is
indistinguishable from double-precision round-off and is dropped by the probe's
own control. (The counts are **upper bounds**: the reach is one global maximum
applied to every cell, which overstates. That is the safe direction for the
1e-14 zero.)

So: **the rectifier is identified, but not at the amplitude that produced the
original puzzle.** It rectifies from about 1e-12 upward.

**The bridge — PLAUSIBLE, not measured.** The ensemble nudge does not stay at
1e-14. NEMO's response reaches ~1e-6 K by day 10, around 1e-7 in relative terms,
*above* the top rung of this ladder where 11% of the ocean is within tipping
reach. The nudge needs only days to grow into the switch's range, and from then
on every further difference is rectified. Consistent with everything measured;
not yet a measurement. One caveat on it: a grown perturbation is spatially
concentrated, so its reach is not directly comparable to a random nudge's.

## Why this fits the original puzzle

The open item was: legoESM's ensemble spread is ~3000× NEMO's at day 90, the
offset appears early, growth *rates* match. A discrete switch produces exactly
that shape — a one-off multiplication when it fires, no change to any growth
rate afterwards.

**But this probe tested legoESM, and the 309× was measured on NEMO.** Both
models carry a convective-adjustment switch, so the mechanism transfers as a
hypothesis. It is not a measurement of NEMO.

## If it is this switch, the quantity that matters is *occupancy*

This trigger was previously shown **population-exact** against NEMO at matched
states — the two models agree on every firing decision. That constrains
decisions at *one* state and says nothing about how many places sit *within a
nudge* of flipping. Two models can agree on every decision and still differ in
how often a disturbance overturns one.

Measured here, legoESM at the shared state: **14.8% of the ocean's interfaces
are convecting** (53,654 of 362,180). **NEMO's half of that census was not
measured.** That comparison — how often each model's switch gets flipped — is
what would connect this to the southern-basin transport deficit, because a
difference in *firing rate* is a difference in time-averaged mixing and so in
the mean state. **No such claim is made here.**

## Reading the table honestly

* **Why an adjacent pair rather than the whole ladder.** The narrow-ramp arm's
  full-ladder spread is still 524 at one step, driven **entirely** by the 1e-8
  rung. That nudge moves stratification by 3.7e-10 — about 3,700× the ramp's
  own width — so at that amplitude the ramp is still, in effect, a switch. The
  1e-12/1e-10 pair is the range where the sharpened arm is genuinely smooth,
  which is why it is the statistic quoted. Both rungs and the full ladder are
  printed; nothing is hidden.
* **The long-horizon columns changed meaning.** An earlier revision read the
  shipped card as *proportional* at 100 and 320 steps (1.95, 2.59). That was an
  artifact of re-deciding the round-off exclusion per column: the control's own
  response grows ~124× over 320 steps and swallowed honest rungs. With the rung
  set fixed at the shortest horizon, the shipped card is rectified at **every**
  horizon (1540, 1869). Corrected.
* **`off` is not a clean control.** Removing convective adjustment changes the
  trajectory itself, which is why that arm becomes rectified again at longer
  horizons. The narrow-ramp arm is the one-variable comparison.
* **The ramp is centred on zero stratification while the hard test fires at
  −1e-12**, so the trigger location moves by 1e-12 — negligible against real
  stratification, but it is a difference and it is stated.
* **One seed, one state, one nudge direction.**

## What to do next

1. **Re-run the neighbourhood census on the *grown* perturbation** — at day 5
   and day 10, not day 0 — on **both** models. That turns the plausible bridge
   into a measurement and produces the firing-rate comparison in the same pass.
   Both states are already on disk.
2. **Then, and only then**, ask whether a firing-rate difference has anything to
   do with the basin deficit.

Not worth doing: further work on the advection limiter or the eddy scheme as
rectifiers. Both are cleanly exonerated at all four horizons.
