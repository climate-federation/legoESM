# What makes legoESM's response wildly out of proportion to a tiny nudge?

**Answer, in one sentence.** Convective adjustment is the culprit — it flips
vertical mixing between 1e-5 and 100 m²/s, a factor of **ten million**, the
instant a column tips unstable — and once the comparison is made properly, the
**sharpness of that edge alone** accounts for a factor of **294** in how far the
model's response departs from proportionality.

But read the two caveats before quoting any of it: the **pre-registered rule
identifies nothing**, and the effect is **silent at the nudge size the
ensembles actually used**.

Pre-registration: `scripts/validate/ocean_fidelity/dino_1226/PREREG_switch_rectifier.md`
with three amendments, all recorded before the numbers they govern. Probe:
`switch_rectifier.py`.

---

## What was measured

From the shared NEMO day-180 ocean state, in double precision, in memory. Nudge
temperature by a relative amount ε, run, record the **gain** — response per unit
of nudge:

```
gain = ‖ T_nudged − T_control ‖ / ε
```

A model responding in proportion gives the same gain for every ε. Five nudge
sizes 1e-16 … 1e-8, four horizons from one step to day 10, seven arms.

## First: what the pre-registered rule says

The registered rule was *"the rectifier is the arm whose disabling brings the
gain spread below 3 at the horizons where the shipped card exceeds it."* The
shipped card exceeds it at **all four** horizons (2031 / 866 / 1540 / 1869).
Measured:

* removing convection: collapses at one step only (1.58), then 10.6 / 46.6 / 33.6
* smoothing the edge: collapses at ten steps only (1.83), else 524 / 52 / 369

**No arm collapses everywhere the card is rectified, so under the registered
rule no arm is identified.** That is the honest pre-registered answer and it
comes first.

## Then: the post-hoc statistic, and what it shows

Everything below was chosen **after** seeing the data and is labelled post-hoc.
The full-ladder spread mixes two effects, so the cleaner cut is the single step
where the shipped card jumps — the gain at nudge 1e-10 divided by the gain at
1e-12, after **one timestep**. 1.0 means proportional.

| arm | what changes | ratio |
|---|---|---:|
| **shipped card** | nothing | **2030.8** |
| advection limiter off | limiter gone, same flux | 2034.4 |
| eddy scheme off | GM/Redi gone | 2017.5 |
| convection off | process gone entirely | 1.0026 |
| convection, trigger moved to zero (**hard** edge) | trigger location only | 295.9 |
| convection, trigger moved to zero (**smooth** edge) | + edge softened | **1.0063** |

**Two clean readings.**

1. **Convective adjustment owns it.** The limiter and the eddy scheme leave
   2031 untouched — exonerated at every horizon. Both convection arms move it
   by three orders of magnitude.
2. **The edge, compared properly, is worth 294×.** The last two rows differ in
   **one thing only** — whether the switch is a step or a ramp — because both
   trigger at the same place. 295.9 → 1.0063.

**Why the last two rows and not "shipped card vs smoothed".** The model's
smoothing branch is centred on zero stratification and **ignores the trigger
offset**, while the shipped hard branch fires at −1e-12. So a smoothed arm
differs from the shipped card in *two* ways at once, and 0.577% of the ocean's
interfaces sit exactly in the band they disagree about — each able to move by a
factor of 1e7. Adding a hard-edged arm at the *same* trigger location is what
makes the comparison one-variable. Splitting the 2031 that way: relocating the
trigger accounts for 6.9×, softening the edge for 294×.

## Retractions — three, all mine

1. **An earlier version claimed the same conclusion from a smoothed arm at the
   default ramp width.** That arm is invalid: its ramp is 1e-6 wide in
   stratification, which is ordinary deep ocean — 40.4% of interfaces sit inside
   it, median mixing moves **31,000×**, 77.3% move by more than 10%. A different
   ocean, not a softened switch. The probe now measures this rather than
   assuming it.
2. **The narrow-ramp arm was then called "one variable" — also wrong**, for the
   trigger-offset reason above. Fixed by adding the matched-trigger arm.
3. **The claim that both quoted rungs are "small enough not to outrun the
   ramp" is wrong by arithmetic.** A 1e-10 nudge moves stratification by
   3.673e-12, i.e. **37×** the 1e-13 ramp width. Both rungs outrun it; only the
   degree differs. The pair is an empirical observation, not a safe bracket.

Also corrected: the pre-registration understated the mixing jump by 100× (it
quoted the software default, not the card), and the long-horizon columns
previously read "proportional" only because the round-off cut was re-decided
per column while the control's own response grows ~124× over 320 steps.

## The catch, and it is load-bearing

**At the nudge size the ensembles used, this switch cannot fire.**

| nudge | interfaces it could tip | share |
|---|---:|---:|
| **1e-14** (the ensembles' size) | **0** | **0%** |
| 1e-12 | 8 | ≤0.002% |
| 1e-10 | 2,773 | ≤0.77% |
| 1e-8 | 41,370 | ≤11.4% |

Counts are **upper bounds** (one global maximum reach applied to every cell),
which makes the 1e-14 zero safe. The 1e-14 rung is also indistinguishable from
double-precision round-off and is dropped by the probe's own control. The first
rung with a real population is 1e-10.

**Honest scope: a rectifier at ≳1e-10 in legoESM, silent at 1e-14. On this
evidence it does not explain NEMO's recorded 309×.**

The bridge that would connect them — the nudge grows into the switch's range
within days, since NEMO's response reaches ~1e-7 relative by day 10 — remains
**plausible and unmeasured**, and a grown perturbation is spatially
concentrated, so its reach is not directly comparable to a random nudge's.

Also: this probe tested **legoESM**; the 309× was measured on **NEMO**. Both
carry such a switch, so the mechanism transfers as a hypothesis only.

## If it is this switch, the quantity that matters is *occupancy*

The trigger is already known **population-exact** against NEMO at matched
states — the two models agree on every firing decision. That constrains
decisions at one state and says nothing about how many places sit *within a
nudge* of flipping. Measured here: **14.8% of legoESM's interfaces are
convecting** (53,654 of 362,180). **NEMO's half was not measured.** That
comparison is what would connect this to the southern-basin deficit, because a
difference in firing *rate* is a difference in time-averaged mixing. **No such
claim is made here.**

## Remaining weaknesses, recorded

* **Rung sets differ across arms** (removing convection keeps four rungs, the
  others three), so full-ladder spreads are not strictly matched across arms.
  The quoted pair is matched — those two rungs survive in every arm.
* **Arm validity is measured at the start only.** The smoothed arm's gains at
  ten steps run 1e6–1e9 against the card's 5e3–1e4: its trajectory diverges. The
  fix needs no new run, only storing each arm's unperturbed control and printing
  the distance between them.
* **At long horizons the matched-trigger arm saturates** — its three smallest
  nudges all produce the *same* response, so those columns measure saturation,
  not proportionality. The one-step column carries the finding.
* One seed, one state, one nudge direction.

## What to do next

1. **Re-run the neighbourhood census on the *grown* perturbation** at day 5 and
   day 10, on **both** models. That turns the plausible bridge into a
   measurement and produces the firing-rate comparison in the same pass. Both
   states are already on disk.
2. **Then, and only then**, ask whether a firing-rate difference bears on the
   basin deficit.

Not worth further work: the advection limiter and the eddy scheme as
rectifiers. Cleanly exonerated at all four horizons.
