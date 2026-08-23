# Does legoESM really amplify a tiny nudge 100× harder than NEMO — and is it the leapfrog?

**Answer, in one sentence.** The amplification is real and *larger* than
published — legoESM's ensemble spread is about **3000× NEMO's** at day 90 — but
it is **not** the leapfrog computational mode: kicking both time levels instead
of one changes the ratio by a factor of **0.96**, against a bar of 4.95, and
the two models turn out to use the **same filter strength**, so there was no
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

**1. The premise is false, and it was checkable offline.** Both models filter
with the **same coefficient, 0.1** — NEMO reports `rn_atfp = 0.1` in its own
output and the legoESM card sets the same value — and neither starts with a
forward-Euler step. The hypothesis needs NEMO's filter to be *stronger*; it is
not.

*Scope of that check, because a reviewer caught the overstatement:* only the
**coefficient** was compared. legoESM applies a thickness-weighted variant of
the filter, not the textbook form, and NEMO's own variable-volume version was
not read from its source. So what is established is that the filter *strengths*
match, not that the two implementations are line-for-line identical. That is
enough for the hypothesis, which is entirely about one side damping harder than
the other.

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

### The calibration did **not** work — and failing it produced the best finding here

The plan was to check the instrument against the filter's own algebra: each
model's spread should grow by 2.25 between the arms. It does not check out, and
an earlier draft of this document claimed it did. **That claim is retracted.**

| | predicted | legoESM | NEMO |
|---|---:|---:|---:|
| between-arm spread ratio, within one model | 2.25 | 1.39 | 1.31 |

With four members, a ratio like this carries a null band of about ±3.1×. So
1.39 is neither distinguishable from 1.0 nor from 2.25. It resolves nothing,
and reading the two models' 6% agreement as confirmation was exactly the error
this run's own decision rule was amended to forbid, pointed the other way.

The design is **paired** — the same seed produces the same draw in both arms —
so the sharper test costs nothing: compare each seed's day-10 perturbation
against its own control, two-level over one-level. In a regime where the
perturbation still grows in proportion to itself, that ratio *is* 2.25, with no
ensemble noise at all. Measured on NEMO, it is **1.00**.

**Why it fails is the finding.** The day-10 perturbations are not proportional
to the nudge at all:

* the six perturbation sizes span a factor of **309**, not a factor of 2;
* only **two distinct locations** carry the largest perturbation across three
  members;
* seeds 1 and 3 — genuinely different runs, from genuinely different perturbed
  restarts — produce perturbation sizes agreeing to **5 parts in a million**,
  at the **same cell**. Independent random draws cannot do that.

Something discrete is setting the amplitude: in some members a localized
process fires and multiplies the nudge by ~300×, in others it does not. That is
rectification, not growth, and it is the leading remaining explanation for an
offset that appears immediately and leaves growth rates alone. It also means
the filter algebra — a statement about proportional response — was never
testable at day 10, on either model.

None of this touches the refutation: it needed a collapse of ~480× and measured
0.96×, and both arms sit in the same regime, whatever that regime is.

### The published 78–435× is too small — the asymmetry is ~3000×

Scored with this probe, the *recorded* runs give a median ratio of **3051×**
across all eleven metrics — an order of magnitude above the published range's
top end. The re-run at the current revision gives 4827×, a further 1.6×.

The probe is not the difference: on the recorded artifacts it reproduces the
recorded report's own headline metric exactly (289.6 here against 290 there).
But **which** metrics the published 78–435× covered is not recorded anywhere,
so this is a *wider re-measurement*, not a demonstration that the same number
was computed wrongly. Stated precisely: **78–435× is not the range over the
eleven metrics this campaign scores**, and it should not be quoted as the size
of the effect.

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

Re-ranked after the second review and after the day-10 measurement above.

1. **Find the switch.** One step, three nudge sizes (1e-14, 1e-12, 1e-10) in
   double precision: is the response proportional to the nudge? The day-10
   evidence says it is not, so the follow-up is to name the culprit — re-run
   with convective adjustment, the mixed-layer index, and the tracer limiter
   disabled in turn until the ~300× amplification stops. Seconds of compute per
   arm, and it is now the leading candidate rather than one of four.
2. **Compare the two models' early transients.** One legoESM member, 60 days,
   double-precision daily temperature dumps, overlaid on NEMO's. NEMO's largest
   temperature difference *shrinks* from 1.74e-06 K at day 10 to 7.79e-09 K at
   day 60 — a factor 224 — before growing to 1.28e-04 K by day 90, while
   legoESM's only rises. In logarithms that dip is about two-thirds of the
   entire 3000× gap, so this may be NEMO's missing decay rather than legoESM's
   excess growth. It cannot be read today: legoESM's early snapshots are at the
   storage floor.
3. **Kick only the cells NEMO also kicks.** 4.8% of the grid is nudged in
   legoESM and not in NEMO — cells below the sea floor, in a model with a
   documented history of below-bottom values leaking into wet stencils. One
   member with the nudge masked to NEMO's non-zero cells settles it, free.
4. **Look at the scale of the spread.** The spatial spectrum of the day-10 to
   day-30 difference: energy at the two-grid-point scale points at the limiter
   or the dissipation, energy at the eddy scale points at the dynamics.
5. **Does the nudged ensemble's *mean* drift off the control?** This replaces
   the pattern projection the pre-registration proposed for connecting this to
   the basin deficit. A deficit is a difference in the *mean*, so mean drift —
   rectification — is the quantity that would connect them; a pattern
   correlation would light up simply because both fields live in the same
   eddy-active basin.

**Dropped:** sweeping the filter coefficient. The filter's own algebra says the
computational mode is gone by day 10, so it cannot set a day-90 amplitude —
keeping that on the list contradicted the rest of this document.

## Limitations a reader should carry away

* **legoESM's early spread is not measured, it is censored.** Its saved states
  are single precision, and a nudged member differs from its control in **2**
  cells out of 372,528 at day 10 and **8** at day 30 — half a unit and one unit
  in the last place. By day 60 it is 26,691 cells and by day 90 59,768. Day 90
  is fully resolved on both sides; the "first 30 days" half of the original
  claim has no legoESM measurement behind it on either arm.
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
whole model state compared before and after. And the two arms' **unperturbed** controls agree
to **exactly zero** on all eleven metrics — which establishes that the new
option is completely inert when it is not used, and rules out an accidental
change to the model along the way. It does not by itself prove that nothing
else differs for the *nudged* members; that rests on the command lines
differing in one token and on the nudge receipts.

## Review

**Code review — completed before any number was cited.** It found the
model-revision mismatch that would have voided the whole comparison, a
precision gate that had never once fired, a decision rule that turned
"underpowered" into a positive claim, a control threshold set at the size of
the signal it was meant to police, and four defects in the perturbation
receipts. All were applied and the pre-registration was amended with them
before scoring.

**Mechanism review — commissioned at the same time, did not return, and was
re-commissioned after scoring.** It has now reported. It re-derived the filter
algebra independently and confirmed it; it confirmed that the two arms really
do differ, by checking that legoESM's stepping consumes the earlier time level
on the first step rather than discarding it, and that both arms ran in double
precision so the nudge was never quantised away; and it confirmed that the
refutation clears its own bar with two decades to spare.

It also found four overstatements in an earlier draft of this document, all of
which are corrected above: the claim that the instrument had confirmed the
filter algebra (it had not — that is the retraction in the calibration section,
and chasing it produced the day-10 finding), "identical time filter" where only
the coefficient was compared, a published-range comparison that set a new
eleven-metric median against an old range of unrecorded scope, and a control
described as proving more than it does. Its ranking of the remaining candidates
is the one used above, including dropping the filter-coefficient sweep as
inconsistent with this document's own argument.
