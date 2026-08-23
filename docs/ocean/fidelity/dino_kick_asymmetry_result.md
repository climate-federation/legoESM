# Does legoESM really amplify a tiny nudge 100× harder than NEMO — and is it the leapfrog?

**Answer, in one sentence.** The amplification is real and *larger* than
published — legoESM's ensemble spread is about **3000× NEMO's** at day 90 — but
it is **not** the leapfrog computational mode: kicking both time levels instead
of one changes the ratio by a factor of **0.96**, against a bar of 4.95, and
the two models turn out to run the **identical time filter**, so there was no
asymmetry there to find.

Written for a reader who has not followed the campaign. Pre-registration:
`scripts/validate/ocean_fidelity/dino_1226/PREREG_kick_asymmetry.md`, committed
before any member ran and amended (before scoring) with the two adversarial
reviews. Probe: `kick_asymmetry.py`.

---

## What was asked

An earlier result recorded, and did not investigate, that an identical 1e-14
relative temperature nudge produces a day-90 ensemble spread **78–435× larger**
in legoESM than in NEMO, while the two models' *growth rates* match. Something
sets the offset in the first month and then both models grow error at the same
speed.

The proposed explanation: the nudge was applied to only **one** of leapfrog's
two time levels. That is the textbook way to excite leapfrog's *computational
mode* — a sawtooth in time that is not part of the physical solution — and if
NEMO's time filter removed more of it, NEMO would start smaller.

The test: nudge **both** time levels with the same relative draw, on both
models, and re-measure.

## Two things were found before any run finished

**1. The premise is false, and it was checkable offline.** Both models run the
identical time filter — the plain Robert–Asselin form at coefficient 0.1
(NEMO reports `rn_atfp = 0.1` in its own output; the legoESM card sets the same
value) — and neither starts with a forward-Euler step. There is no asymmetry
between the two filters for the hypothesis to live in.

Worse for the hypothesis, the filter's own algebra makes a **point prediction**:
a one-level nudge leaves 4/9 of itself on the physical solution, a two-level
nudge leaves all of it, so switching arms should multiply each model's spread
by **2.25** — the *same* factor on both sides, which therefore **cancels** in
the legoESM/NEMO ratio. The ratio was predicted to move by exactly **1.00**
before a single member was scored.

That turns the experiment from a test into a **calibration**: does the
instrument see the 2.25 the algebra demands, and does the ratio then sit at 1?

**2. The two arms were about to compare two different models.** The plan reused
recorded runs for the one-level arm. Those ran 45 commits back, across two
commits that change the physics of the very configuration under test. They were
re-run at the current revision. The oracle side needed nothing — same certified
executable, and the only namelist difference is when the run stops.

## What was run

Sixteen 90-day runs from the same shared ocean state: four legoESM and four
NEMO per arm, one unperturbed control plus three nudged members each.

## The result

| | one-level nudge | two-level nudge | change |
|---|---:|---:|---:|
| **legoESM spread ÷ NEMO spread** (median over 7 metrics) | **4827×** | **6395×** | **0.96×** |

The bar, fixed in advance: with four members per ensemble, any change smaller
than **4.95×** is indistinguishable from no change at all. The measured change
is 0.96×. The hypothesis needed a collapse of ~480× to bring the ratio into the
range where the two models could be said to wobble comparably. Even the
optimistic end of the measurement's own error bar reaches 4.8×, two orders of
magnitude short.

**Verdict: REFUTED.** Removing the time-level mismatch does not touch the
asymmetry.

### The calibration says the instrument was working

| | predicted | legoESM | NEMO |
|---|---:|---:|---:|
| spread ratio between the two arms, within one model | 2.25 | **1.39** | **1.31** |

Both models show the arm effect, both at about 60% of the day-0 prediction —
expected, since the prediction is about the instant the nudge is applied and
this is measured 90 days and much nonlinear growth later. The load-bearing
number is that the two sides agree with **each other** to 6%. That is precisely
why the ratio between them did not move, and it is the closed form's
prediction confirmed rather than a null result from an instrument that saw
nothing.

### The published 78–435× is too small — the asymmetry is ~3000×

Scored with this probe, the *recorded* runs give a median ratio of **3051×**
across the eleven metrics (the re-run at the current revision gives 4827×, a
further 1.6×). The probe reproduces the recorded report's own headline metric
exactly (289.6 here vs 290 recorded), so this is not an instrument difference:
the published range was quoted from a narrower subset and understates the
effect by about an order of magnitude. **The 78–435× figure is retracted.**

### One thing the two-level nudge *did* change

NEMO's spread used to *shrink* between day 30 and day 60 before growing — the
oddity that motivated the hypothesis. Under the two-level nudge that reversal
disappears on **9 of the 9** metrics that showed it. So the one-level nudge
does put NEMO into a transient the two-level nudge does not, and that transient
is real. It is simply not what sets the 3000× amplitude gap — the amplitude
gap is the same in both arms. The mode that damps out is too fast to be this
transient (it dies by day 10), so what the one-level nudge excites in NEMO is a
slower adjustment that has not been identified. Recorded, not explained.

## What this does and does not mean for the basin deficit

The unexplained southern-basin transport deficit is currently classified as
individually-too-small term differences that some feedback makes visible. A
model that amplifies a perturbation 3000× harder than its oracle looks like
exactly that kind of feedback — and the pre-registration originally said a
refutation here would promote it to a leading candidate.

**That claim was withdrawn before scoring, and is not made.** Ensemble spread
measures how fast two nearby trajectories separate; the deficit is a difference
between two settled climates. They connect only if the amplifying pattern
actually overlaps the deficit pattern, and that has not been measured. Both
fields are already on disk, so the measurement is free — it is named below, not
done here.

## What is now worth doing, cheapest first

1. **One step, three nudge sizes** (1e-14, 1e-12, 1e-10) in double precision:
   is legoESM's response proportional to the nudge? If not, a discrete switch —
   convective adjustment firing, a mixed-layer level index moving — is
   rectifying the nudge in a single step, which would explain an offset that
   appears immediately and leaves growth rates untouched. Seconds of compute,
   and it is the single best remaining candidate.
2. **One legoESM member, 60 days, double-precision daily temperature dumps**,
   overlaid on NEMO's. NEMO's raw perturbation *shrinks* 224× between day 10
   and day 60 before growing; whether legoESM shares that dip is one curve and
   it settles the "born in the first 30 days" claim.
3. **Sweep the filter coefficient** on one model. Insensitivity kills the
   filter family of explanations with one variable and one model.
4. **Project** the day-90 spread pattern onto the basin-deficit pattern — the
   measurement that would license, or kill, the feedback connection above.

## Limitations a reader should carry away

* **legoESM's early spread is not measured, it is censored.** Its saved states
  are single precision, and between day 10 and day 60 the nudged members differ
  from their control in a handful of cells out of 372,528 — at the storage
  resolution. Day 90 is fully resolved on both sides; the "first 30 days" half
  of the original claim has no legoESM measurement behind it on either arm.
* **Four members.** Every spread is a factor-of-two estimate and every ratio a
  factor-of-four one. That is why the bar is 4.95 and why a 0.96 is read as
  "nothing moved" rather than as a precise 4% shift.
* **Four of eleven metrics were dropped** from the median because their spread
  sits at or below the single-precision storage resolution — they were
  measuring the file format. The gate that was supposed to catch this had never
  fired in this campaign: it compared an already-single-precision state against
  itself and always returned zero. Fixed at the source, which means the same
  flag was inert in the earlier 360-day verdict too.
* **The nudge lands on slightly different cells in the two models** (4.8% of
  the grid is nudged in legoESM and not in NEMO, below-bottom cells where the
  two store different values). Identical in both arms, so the comparison
  between arms is protected; the absolute ratios inherit it.
* **The nudge is on temperature; the scores are transports.** Velocity and sea
  surface height carry a one-level mismatch in *both* arms, so what is refuted
  is time-level content in the tracer equation — which is what was proposed.

## Controls, all green before any number above was read

Both arms' legoESM members at one source revision, and the **same** revision
across arms. Both models confirmed to share the time-filter coefficient, read
from each side's own output. All sixteen runs judged complete by their own
success line, never by an exit code. Every nudge receipted: on the oracle side
every variable and every attribute compared against the source, with the two
time levels confirmed to have received the same draw; on the legoESM side the
whole model state compared before and after. And the decisive one — the two
arms' **unperturbed** controls agree to **exactly zero** on all eleven metrics,
so the arms differ in the nudge and in nothing else.

## Review

Two independent adversarial reviews before any number was cited. The code
review found the model-revision mismatch that would have voided the comparison,
an inert precision gate, a decision rule that turned "underpowered" into a
positive claim, and a control threshold set at the size of the signal. The
mechanism review found that the hypothesis's premise was checkable offline and
false. All findings were applied, and the pre-registration was amended with
them before scoring.
