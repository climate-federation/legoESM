# RECEIPT — VORTEX round 3: the vector-EEN card, and who owns the flux card's second step

Date 2026-09-29. Lane tip `eeef1ec91397`. Two preregistrations, both frozen
before their measurements: `PREREG_nemo_testcases_l1_vortex_round3_vector.md`
(part A) and `PREREG_nemo_testcases_l1_vortex_round3_kt2_owner.md` (part B).

Status: **part A SCORED, part B HELD AND ITS ATTRIBUTION RETRACTED.** Both
adversarial reviewers returned DO NOT SHIP on part B and agreed on the leading
finding; section 12 records what they found and what it costs. Part B's owner
verdict is demoted to PLAUSIBLE and nothing rests on it. Part A is clean on
both reviews. The second VORTEX card exists, runs,
and is scored on its own ten-step ladder. The first card's second-step debt is
attributed to one stage, with three named candidates refuted by measurement and
the remaining walk needing one more NEMO record.

## Part A — the vector-EEN card

### 1. What changed, and nothing else

NEMO's own resolved run is the statement of what this card is: "vector form :
keg + zad + vor is used" and "vector form dynamics : total vorticity = Coriolis
+ relative vorticity". Two deck lines produce all of it.

| namelist entry | flux card (round 2) | vector card (round 3) |
|---|---|---|
| `ln_dynadv_vec` | `.false.` | `.true.` |
| `ln_dynadv_up3` | `.true.` | `.false.` |

`ln_dynvor_een` is unchanged. NEMO counts the advection forms and stops unless
exactly one is selected, and that same count decides what the vorticity routine
is handed: the flux card gets the planetary rotation plus a metric term this
mesh makes bitwise zero, the vector card gets the planetary rotation plus the
live relative vorticity.

| statement | compiled source | lines |
|---|---|---|
| exactly one advection form, and which one it is | `VORTEX_VEC_OMIP_L1/BLD/ppsrc/nemo/dynadv.f90:184-190` | 7 |
| what vector form calls: the kinetic-energy gradient and the vertical advection of momentum | `VORTEX_VEC_OMIP_L1/BLD/ppsrc/nemo/dynadv.f90:135-138` | 4 |
| what the vorticity routine is handed under vector form | `VORTEX_VEC_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:861-864` | 4 |
| the run's own resolved deck | `vortex_round3/namelist_cfg:182-185` | 4 |

The card is built from the flux card's own configuration with only the
momentum fields replaced, so "one variable" is structural here rather than a
claim in prose. Nothing new was written: the arm it selects is the one the
ORCA2 card already runs, which is what the round-2 receipt predicted.

Every untranscribed pairing raises. The proof that the metric term vanishes on
this mesh stays attached to the flux card, because the vector card does not
rely on it — NEMO never forms that term on a vector-form deck.

### 2. The record

The acquisition ran and was ADMITTED: restart byte-identical to the
un-instrumented reference run, 24 records, every one parsed from its own
header. Evidence in `phase3/vortex/round3`.

**The two runs' kt=1 step-entry records are BYTE-IDENTICAL** (prediction P1,
confirmed by `cmp`). The stage-1 record, the barotropic frame and the kt=2
entry all differ, which is the momentum change arriving exactly where it
should. So the initial state is provably shared, and the ladder below is a
momentum comparison and nothing else.

### 3. The initial state

Unchanged from round 2, as P2 predicted, and for the strongest possible
reason — it is the same bytes:

| field | cells unequal | of | worst |
|---|---|---|---|
| temperature | 0 | 37210 | exact |
| salinity | 0 | 37210 | exact |
| zonal velocity | 788 | 36600 | 2 last bits |
| meridional velocity | 788 | 36600 | 2 last bits |
| sea surface height | 104 | 3721 | 2 last bits |

### 4. The two ladders, side by side

Normalized maximum absolute error against the bar `1.0e-15`. Salinity is at
bar on both cards until kt=7 and is omitted.

| kt | T flux | T vector | u flux | u vector | ssh flux | ssh vector |
|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 2.220e-16 | 2.220e-16 | 1.355e-20 | 1.355e-20 |
| 2 | 1.643e-09 | 2.145e-08 | 1.136e-07 | 1.432e-05 | 3.709e-08 | 2.269e-05 |
| 3 | 5.510e-09 | 1.407e-07 | 2.582e-07 | 1.449e-05 | 6.503e-06 | 9.397e-06 |
| 4 | 1.128e-08 | 2.091e-07 | 1.073e-06 | 1.457e-05 | 8.218e-06 | 8.003e-06 |
| 5 | 1.698e-08 | 1.710e-07 | 1.061e-06 | 1.451e-05 | 7.978e-06 | 8.602e-06 |
| 6 | 2.075e-08 | 1.560e-07 | 1.288e-06 | 1.601e-05 | 8.024e-06 | 7.854e-06 |
| 7 | 2.154e-08 | 1.824e-07 | 1.834e-06 | 1.636e-05 | 5.160e-06 | 5.332e-06 |
| 8 | 2.028e-08 | 2.206e-07 | 2.167e-06 | 1.548e-05 | 5.950e-06 | 6.077e-06 |
| 9 | 2.075e-08 | 1.843e-07 | 2.360e-06 | 1.451e-05 | 4.537e-06 | 4.696e-06 |
| 10 | 2.103e-08 | 2.014e-07 | 2.464e-06 | 1.462e-05 | 5.336e-06 | 5.333e-06 |

**First over the bar: kt=2 on both cards**, on temperature, both velocities
and sea surface height together; salinity survives to kt=7 on both.

Read plainly: the vector card enters the ladder at the bar and leaves it at
the first step it takes, like the flux card, but it leaves by **126 times
more** on velocity and **612 times more** on sea surface height. Its error
then stops growing — flat at about 1.5e-05 from kt=2 to kt=10, where the flux
card's velocity error grows twenty-fold over the same span. A large error that
does not grow is a fixed transcription difference being re-made every step,
not an instability.

Predictions P1-P5 all held. P6 is section 6.

### 5. What the vector card's size means, stated carefully

The vector card runs three operators the flux card does not: the
kinetic-energy gradient, the vertical advection of momentum, and the vorticity
routine on the live relative vorticity. Its residual is two orders larger than
the flux card's. That is a RANKING, not an attribution: it says the
vector-invariant momentum set carries more debt on this geometry than the
flux-form set does, and it does not say which of the three operators carries
it. Naming one here on the strength of a magnitude ordering is the exact
mistake this campaign has paid for before; round 4 measures it.

One thing the pair does settle, because the two runs share their initial state
byte for byte: whatever owns the vector card's debt is in the momentum program,
not in the geometry, the equation of state, the tracer program, the free
surface or the initial state, all of which are identical between the two cards.

## Part B — who owns the flux card's second step

### 6. The walk

Round 2 measured the flux card's ladder and named nobody. The walk substitutes
one boundary at a time from NEMO's own record of the inside of the first step —
the barotropic frame written immediately after the external solve, and the
states written at each stage's pointer boundary — and scores the kt=2 entry
with the same normalized maximum the ladder gate reports.

| arm | what is substituted | T | u | ssh |
|---|---|---|---|---|
| 0 | nothing (the card) | 1.643e-09 | 1.136e-07 | 3.709e-08 |
| 1 | NEMO's kt=1 entry state | 1.643e-09 | 1.136e-07 | 3.709e-08 |
| 2 | + NEMO's external-solve handoff | 1.643e-09 | 1.136e-07 | **0** |
| 3 | + NEMO's stage-1 output as stage 2's entry | 1.654e-09 | 1.177e-07 | 0 |
| 4 | + NEMO's stage-2 output as stage 3's entry | **3.466e-16** | 1.204e-07 | 0 |

Four findings, each a measurement:

**The initial state owns NONE of it.** Arm 1 reproduces arm 0's temperature
and both velocities BIT FOR BIT, and its height row differs only in the last
three digits (`3.7087937716461764e-08` against `3.708793771648887e-08`). The
one-to-two-last-bit residual on the initial velocity and height is exonerated;
round 2's "not worth closing" verdict stands. (Prediction B1 confirmed.)

**The external solve owns the sea surface height row OUTRIGHT.** Substituting
NEMO's handoff puts the height at exactly zero, and leaves the velocity
untouched. So on this card the end-of-step height is the external solve's own
output and nothing else writes it. (B2 confirmed.)

**THE PICK WAS WRONG, AND IS RETRACTED.** B3 predicted the external solve would
carry the majority of the momentum residual too, and it carries none of it:
arm 2's velocity is arm 1's to the last digit. The reasoning behind the pick —
that the rotation is new, enters twice, and already owns the height row — was
plausible and was worth nothing.

**The owner is the stage-3 momentum update.** Arm 4 hands stage 3 NEMO's own
entry for everything it consumes, and stage 3 still produces the whole
residual, 1.204e-07 — more than all three stages together produce in arm 2, so
stages 1 and 2 partially cancel it. In the same arm the temperature goes to
3.466e-16, at the bar: stage 3's TRACER program is exact given NEMO's entry,
and its momentum program is not. (B4's stage ranking is therefore moot and is
withdrawn: stage 3's own share is the whole of it.)

### 7. Inside stage 3 — THIS SECTION IS RETRACTED

**Everything below this line about the vertical viscosity was produced by an
UNCOMMITTED probe and is therefore unmeasured.** Both reviewers found it
independently and it is the campaign's own named failure mode: a throwaway
probe's number is not evidence, and the committed walk computes no viscosity
increment, no ablation and no correlation. The numbers are kept below, struck
through in words rather than deleted, so the next round knows exactly what to
re-measure with a committed instrument; NO decision may rest on them, and the
implicit vertical viscosity is BACK on the candidate list for round 4.

### 7 (retracted). What the residual was claimed not to be

Two further measurements, both refutations, both worth more than the
hypotheses they killed.

**It is not the depth mean.** The residual's per-column mean is 6.8e-17 —
zero. Whatever is wrong leaves the barotropic component exactly right, which
is what NEMO's own barotropic correction at
`VORTEX_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:412-418` forces. The error is
entirely in the vertical PROFILE, and it is surface-weighted: 1.20e-07 at the
top level, through a sign change around the fourth, then flat at about
4.6e-08 through the bottom half of the column.

**It is not the implicit vertical viscosity**, which was the leading suspect
on magnitude. Stage 3 is the only stage that runs it
(`VORTEX_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:405`), and its own increment on
this card is 2.298e-07 — comfortably large enough to carry a 1.2e-07 residual,
which is exactly why magnitude alone is not evidence. Correlating the residual
against that increment cell by cell gives **-0.0018**: no relationship at all.
For contrast, the same correlation against the arm with the viscosity removed
is -0.889, so the instrument can see a relationship when there is one. The
tridiagonal at `VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynzdf.f90:191-195`, the
thickness-weighted update at `:143-147` and the barotropic removal at
`:159-161` are therefore not where the walk continues.

**A candidate checked and refused before it became a claim.** NEMO's
`ln_drgimp` resolves TRUE on this deck even though the drag is off, so the
card's selection of the branch that removes the barotropic velocity before the
implicit solve is right, not a defect. It was checked in the run's own resolved
output before anything was written about it.

**What is left, and why it needs a record.** Stage 3's momentum program
differs from stages 1 and 2 in a short list: the full time step rather than a
third or a half of it, the vertical viscosity (refuted), and the surface-height
ratio at the vertex, which stage 3 forms as the mean of the before and after
values (`VORTEX_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:239`) where the earlier
stages use a two-thirds/one-third blend. Reading that statement against
legoESM's, the two agree algebraically and would differ only in the last bits,
which is eight orders too small — so the stage-3 divergence is most likely a
statement stage 3 SHARES with the earlier stages, whose error only survives
because stage 3's output is the step's output. Separating those needs NEMO's
per-term momentum tendency at a stage, which this record does not contain.
That is round 4's acquisition, and no fix lands until it exists.

**Nothing was landed, and the owner verdict is PLAUSIBLE, not measured.**
Prediction B5 said the owner would be a transcription
difference rather than a rounding floor; that remains the standing hypothesis
and is NOT confirmed, because no statement has been named. Labelled PLAUSIBLE.

## 8. Additivity, and one red that is not this round's

**The VORTEX flux card and both tanks are unchanged** through the same gate,
re-run at this tree. See section 9.

**A pre-existing red, reproduced with this round's work removed.** The GYRE
card's configuration digest no longer matches the value round 2 pinned:
`eaef11b4c2e37a31` expected, `337651dbd9f1b49c` measured. This was reproduced
on a tree with every round-3 edit reverted, so it is NOT this round's. The
owner is the 2026-09-29 merge of GitHub main: that merge flipped the ocean
biharmonic mixing's `enforce_cfl` default from false to TRUE and removed its
companion estimate field, and GYRE is the only certified card whose
configuration carries a lateral-mixing block for those fields to appear in.
GYRE's biharmonic coefficients are both exactly zero, so the change is
numerically inert on that card — but a default that moves under a certified
card is a finding, not a formality, and the pin was deliberately NOT re-set
here: re-pinning a digest without proving what moved is what the test's own
docstring forbids, and the decision of whether that default belongs on this
lane is the user's.

That same default flip is ALREADY an open item in the merge receipt
(`nemo_testcases_l2_gyre_merge_main_2026-09-29_receipt.md`), which reported it
as pre-existing on main and made the separate point that it now makes any
lat-lon card selecting the factory biharmonic mixing raise. What is new here is
only that it also moved a certified card's configuration digest, which that
round's gates did not run. So this is one finding with two consequences, not
two findings, and it stays reported rather than fixed.

## 9. Gates

| gate | its own success line |
|---|---|
| acquisition preflight, vector variant | `PREFLIGHT_OK variant vec`, exit 0 |
| acquisition preflight refusals | vector variant handed the flux deck: exit 67; unknown variant: exit 64 |
| record admission, vector card | `ADMITTED`, restart byte-identical, 24 records parsed from their own headers |
| admission plant | refused, exit non-zero |
| kt=1..10 ladder, vector card | scored; section 4 |
| ladder plant, vector card | first-over-bar forced to kt=1, exit non-zero |
| kt=1..10 ladder, flux card | unchanged, every row identical to round 2 |
| kt=1..3 ladder, both tanks | unchanged |
| kt=2 walk | five arms; section 6 |
| kt=2 walk plant | exit 1 |
| walk record readers, three planted malformations | `5 passed` |
| card test and constructibility, plus the recipe, barotropic-state and stage-face-mask tests | `1 failed, 80 passed in 996.00s` — the ONE failure is the pre-existing certified-card digest red of section 8, reproduced with every round-3 edit reverted |
| citation gate, this receipt | `PASS`, 10 citations, 0 unmapped, 0 map entries failing |
| citation gate, round 2's receipt and the lane's own | `PASS` on both, unchanged in count |
| citation gate, planted shift | exit 2 |
| DINO month gate (note BI; this round changes `packages/`) | `2.040288957e-03 K against bar 2.244317642e-03 K (certified 2.040288765e-03 K) -- PASS` |

**The DINO month gate needed a second run, and the reason is worth recording.**
Its first run finished the thirty-day model integration and then REFUSED at the
scoring step, because the gate writes into one shared directory and another
session's output from that morning was already sitting in it. Nothing was
deleted; the gate was re-run against a private directory under this round's own
evidence folder, which is why the number above can be attributed to this tree at
all. The gate as written cannot be run by two sessions at once, and until it
takes a per-caller directory it has to be run alone. The value is
`1.9e-10 K` from the certified one, which is the harness's own run-to-run floor
(round 129 measured about `2e-10 K`), so this round is INERT on DINO — neither
an improvement nor a regression, which is what an additive round should give.

## 10. Choices made this round

| choice | ASKED or UNASKED | note |
|---|---|---|
| the vector-EEN card exists, with these two switches | ASKED (decision 73) | note BJ names both switches |
| its own build, own record, own ladder, kept as a permanent gate | ASKED (decision 73) | |
| the dead flux-form momentum scheme selector on the vector card carries the same inert value the GYRE and ORCA2 vector cards carry | UNASKED, and stated | the field has no "unused" value; the alternative is leaving the OTHER card's selector in place, which is worse. One line for the operator: keep it, or add an explicit "not selected" value to that field? |
| the acquisition script grew a variant selector rather than a second copy | UNASKED, and stated | it is what makes the two records differ by one deck hunk and nothing else |
| the GYRE digest pin was NOT moved | UNASKED, and stated | see section 8; moving it would hide a default that changed under a certified card |
| nothing was landed for the kt=2 debt | ASKED in effect | the preregistration says the round holds if the owner is named without a proven fix |

## 11. OPEN

**Round 4 — the resolution ladder (decision 74, note BK).** Both cards at 15 km
and 10 km without nesting, each with its own record and ladder, one receipt
with the three resolutions side by side. Note that decision 74 says "once both
cards are at bar at 30 km", and NEITHER is: both leave the bar at the second
step. The ladder is still worth running and is arguably worth more now — if
the residual scales with the grid it is truncation, and if it does not it is
the transcription gap both cards are carrying.

**Round 4 or 5 — the stage-3 momentum acquisition.** Naming the statement
behind the flux card's second step needs NEMO's per-term momentum tendency at
a stage, which no existing record holds. Until it exists, no fix lands.

**The vector card's own owner is unnamed**, deliberately. It is 126 times the
flux card's, it does not grow, and the two cards share their initial state byte
for byte, so it is in the momentum program. Which of the three operators
vector form adds is measured, not guessed.

**Carried from round 2:** whether the split-explicit barotropic arm should have
been its own round, and the mesh guard being stricter than the source needs.
**Carried into this round:** the GYRE digest pin (section 8).


## 12. Reviews

Both reviewers were given the whole diff and told to refute it. Both returned
DO NOT SHIP on part B, and both independently led with the same finding, which
is the strongest signal either could have given.

| reviewer | verdict | its leading finding |
|---|---|---|
| codex, adversarial, read-only | DO NOT SHIP | the receipt's viscosity magnitude and correlations exist only in prose; the committed walk computes none of them |
| a fresh Claude code-reviewer | DO NOT SHIP (request changes) | the same, named as the campaign's own uncommitted-probe rule |

Both PASSED part A. Codex: "the vector card correctly changes only the five
legoESM selectors implied by NEMO's two namelist switches", and "no certified
card or shared production dynamics path changes in this range". The Claude
reviewer independently hand-diffed both deck patches against the shipped
namelist and reached the same conclusion, and independently confirmed that the
GYRE digest red is pre-existing at the base commit.

Findings, and what each costs:

| finding | disposition |
|---|---|
| the viscosity numbers come from an uncommitted probe | RETRACTED, section 7. The viscosity is back on the candidate list |
| the walk's plant only perturbs the first arm's scoring, so it passes even if both substitution hooks are inert | ACCEPTED, NOT FIXED. The attribution is demoted to PLAUSIBLE until a plant perturbs a substituted arm |
| arm 1 does not substitute the carried barotropic velocity pair, which the first stage consumes | ACCEPTED. "The initial state owns none of it" is weakened to "the initial state's TRACER, VELOCITY and HEIGHT fields own none of it"; the carried pair is untested |
| arm 2 omits the recorded pre-stage momentum right-hand side, which the instrument does dump | ACCEPTED. "The external solve owns the height row outright" stands for the height (arm 2 puts it at exactly zero, which no omission can fake); the velocity half of that claim is weakened to "substituting the external solve's velocity outputs alone does not move it" |
| the walk always reports MEASURED and exits zero, so it has no verdict of its own | ACCEPTED, NOT FIXED. It is a measurement tool this round, not a gate; it must gain a verdict before anything lands on it |
| the commit range is five commits, not six | corrected |

None of this changes part A, and none of it changes a number in the two
ladders: the walk touches no model code and nothing was landed.
