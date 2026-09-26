# The DINO wall 2-step surface flicker: a launch transient, and a sustained excess it was hiding

Branch `fidelity/dino-wall-flicker`, off `fix/twin-euler-start-default`.
Campaign #1455 / #1226. Two adversarial reviews, two rounds each; the second
round found a result the first draft's aggregate had masked, so the verdict
below is in two parts and only one of them is a refutation.

## Verdict

The two-step (2 x 2700 s) surface mode reported at **2.9x NEMO's amplitude
with 85% of it on land-adjacent rows** is two different things added together,
and the wall has to be split by DIRECTION to see it. DINO's wall band is 320
**meridional** cells (land to the east or west — the north-south basin
sidewalls) plus 100 **zonal** cells (land to the north or south — the first
wet rows against the closed end walls at about +/-69.5 deg).

**(a) On the meridional walls: a launch transient, and it cancels.**
6.60x over the first eight samples, **0.95x** by the last half (interval
[0.75, 1.20]), locus collapsing from enrichment 23.9 to 3.9. It does not
survive into a member-minus-member difference (enrichment 0.90 there — it is
DEPLETED). Nothing to fix and nothing downstream.

**(b) On the zonal walls: a SUSTAINED excess that survives everything.**
Last half **2.68 / 2.85 / 7.59** under the three estimators, interval
**[1.26, 5.10] excluding unity**, both sides clearing their own window-matched
floors (legoESM 2.7x, NEMO 4.6x, with 86% and 88% of individual samples
clearing) — and it repeats at a second start state ten days later (2.86, while
the meridional band there reads 0.96). Its locus does the OPPOSITE of the
transient's: legoESM's zonal enrichment RISES from 0.44 at the first sample to
**17.1** by the last half, against NEMO's 3.5. And it is enriched **8.6x** in
the member difference where the meridional transient is depleted.

**Damping is equal on that sub-band**, so equal damping plus a sustained 2.7x
level at two states means a **source difference on the zonal walls**. The
brief's hypothesis 1 is therefore **not refuted — it is RELOCATED** (zonal,
not meridional) and reduced about 2.5x from the published size. Hypothesis 2
is refuted.

**No suspect is named, and the discriminator was RUN rather than proposed —
but it disqualified most of its own channel.** A term-by-term momentum budget
on those rows, nine matched states, is valid in the two-step band for exactly
one term: lateral friction, once a time-level defect in legoESM's shared
diagnostics was found and fixed. There it is a clean null (the difference
carries 1.1e-5 of the oracle's own two-step content). For the other three
terms legoESM's tendency and NEMO's trend differ by a factor of exactly 2.000
in that band — identically on a second, unrelated wall — so they are different
objects and no source statement can be made from them. The steady channel is
unaffected and shows no term that is systematically worse at the wall.

The live candidates are therefore the barotropic split-explicit machinery,
which this budget cannot see at all, and the **explicit-versus-implicit wind
stress placement** at a wind-driven wall row. Both suspects proposed before the
budget were refuted outright (see the retractions).

**Geography worth flagging, and it is not the reviewers' claim or mine:** the
southern zonal wall row sits at about 69.5 deg S — the same wall row as the
campaign's southern-basin transport deficit.

**The directional split is POST HOC and travels with the result.** Nothing
pre-registered it: the pre-registration covered the aggregate wall band, the
second start state, the turbulence-bridge arm and the member difference, and
the split was proposed by review AFTER those numbers were seen. What that
costs, stated rather than buried: with two candidate sub-bands, an interval
excluding unity at one of them is weaker evidence than a registered test of
the same size. What it does not cost: the split is not a search over many
partitions — DINO has exactly two kinds of wall and they are named by the
geometry, not by the data — and the same split reproduces at the second start
state, in the member difference, and in the locus. The honest reading is a
strong lead that has not yet had its own registered test, and the hand-off
below is what that test should be.

## Retractions

Each was caught by review, not by me.

1. **"legoESM ships `barotropic_forcing_centred = False` against NEMO's
   `ln_bt_fw = .FALSE.`" is FALSE and WITHDRAWN.** The card under test sets it
   **True** (`packages/ocean/legoesm/ocean/experiments/dino.py:1582`, inside
   `DINO_RECIPES["nemo_dino_kamm_mlf"]`, verified in the worktree the run
   used). I read the dataclass default and the forward-Euler base card's
   comments — neither is the resolved configuration. There was never a
   mismatch to suspect.
2. **"Not an initial-condition transfer gap" was WRONG ON ITS FACTS.** The
   turbulence state is not carried over from NEMO's restart. I registered a
   rule and ran it: bridging it moves the wall transient by **0.99x**, so the
   cold turbulence start is refuted as its owner — but the conclusion survives
   only because it was measured, not because the premise held.
3. **"legoESM is if anything LOWER late" is WITHDRAWN.** It subtracted a floor
   measured over the launch transient from a last-half amplitude. Both
   reviewers found it independently. The floor now takes an explicit window
   and the subtraction is dropped.
4. **"Equals NEMO to 3%" is WITHDRAWN as spurious precision**, and so is the
   whole aggregate framing that produced it — see the narrowings.
5. **"The oracle ends up MORE wall-enriched than the twin" is WITHDRAWN as an
   aggregate.** It averaged 320 matched cells with 100 unmatched ones. Per
   direction: NEMO is more enriched on the MERIDIONAL band (5.49 against 3.85)
   and much LESS on the zonal one (3.49 against **17.1**). The meridional half
   is itself thin — half of NEMO's variance there sits in 4 of 320 cells.
6. **"The flicker cannot be the differential noise source" is NARROWED to the
   LAUNCH TRANSIENT only.** The member-difference measurement licenses that
   for the meridional transient (depleted, 0.90). The zonal band is enriched
   8.6x in the very same difference field, so nothing here excuses it.
7. **"Rules out the free-slip domain edge outright" is WITHDRAWN.** DINO's
   outermost j rows being dry rules out the dry HALO row. It says nothing
   about the first WET row — which is exactly where the sustained excess sits.
8. **"Ringdown identical to 0.1%" is WITHDRAWN as over-precise.** See below.
9. **The day-190 "CONFIRMS" is downgraded to "not inconsistent with"**: its
   pre-registered thresholds sat inside the estimator's own noise.
10. **The budget's first null is WITHDRAWN**, twice over: it reduced to a band
    mean before taking the two-step component (which cancels a spatially
    signed source by 44x to 80,000x), and it compared lateral friction across
    time levels. Both are fixed; the corrected result is narrower, not
    broader. The claim that "the differences are steady offsets" was FALSE for
    lateral viscosity, whose apparent time-noise was entirely the time-level
    defect.
11. **I over-claimed novelty.** That the aggregate mode decays to 1.1x late was
    already published (`dino_eta_wave_field_result.md:649`).

## The measurements

Instrument: `scripts/validate/ocean_fidelity/dino_1226/eta_flicker_decay.py`
on the campaign's `eta_wave_twin.py`, which this branch extends with one
shared weighted-rms reduction (bit-identical to the published one — a test
pins it), a region- and window-aware leakage floor that can also return the
PER-SAMPLE series, and `wall_direction_partition`. 22 unit tests, ruff clean.

Everything below is per-step data; DINO's timestep IS 45 minutes, so a
two-step mode sits exactly at the series' Nyquist frequency and is fully
resolved with no rerun.

### Amplitude by direction, day 180 (160 samples, both models from one restart)

| band | cells | area | first 8 | last half (3 estimators) | interval | floors cleared |
|---|---|---|---|---|---|---|
| meridional wall | 320 | 3.56% | **6.60x** | 0.95 / 1.08 / 1.11 | [0.75, 1.20] | 3.0x, 3.4x |
| **zonal wall** | 100 | **0.24%** | 4.24x | **2.68 / 2.85 / 7.59** | **[1.26, 5.10]** | 2.7x, 4.6x |
| aggregate wall | 424 | 3.81% | 6.59x | 1.03 / 1.21 / 1.33 | [0.79, 1.32] | 3.0x, 3.5x |
| interior | 9256 | 91.7% | 2.23x | 1.16 / 1.31 / 1.72 | [0.80, 1.56] | 3.2x, 4.3x |

Day 190, the pre-registered second start state (one variable — same NEMO
binary by md5, same namelist but three step counters, same probe, same mesh;
the legoESM arm ran the day-180 arm's exact model code with a one-line change
to the restart filename): meridional last half **0.96** (interval
[0.48, 1.75]), zonal **2.86** (interval [0.99, 9.72]). The zonal interval at
this state only just touches unity and is very wide — 64 samples against 160,
and a 100-cell band. It is a consistent second observation, not a second
independent confirmation, and the day-180 interval is the one that excludes
unity.

Intervals are 90% moving-block bootstraps with the block length MEASURED from
each series' own decorrelation lag rather than chosen, and with the two models
resampled on INDEPENDENT indices — they are independent trajectories after
step 1, and common indices impose a pairing the data does not have.

The third estimator (median cell) is not area-weighted while the other two
are; on a grid spanning 70S-70N that makes it a different question — "the
typical cell" rather than "the typical square metre" — and it is the one least
sensitive to a few hot cells. Which matters here: half of NEMO's late zonal
variance sits in ONE of its 100 cells against four for legoESM.

### Damping, on a common excitation

Each model minus its OWN free run, so both ring down from an identical imposed
bump — the only lane in which decay rates are comparable at all.

| band | decay constant, NEMO | legoESM | ratio | r² | window range | samples clearing their floor |
|---|---|---|---|---|---|---|
| meridional | 5.06 steps | 5.06 | 0.9993 | 0.967 | 2.7 - 8.5 | 14% |
| **zonal** | 14.10 steps | 14.10 | **0.9999** | **0.793** | 8.3 - 18.9 | 43% |
| interior | 4.18 steps | 4.18 | 0.9994 | 0.986 | 4.0 - 10.0 | 29% |

**Read these honestly, which the first draft did not.** The ratio's four-digit
agreement is not four-digit precision on a physical quantity: both sides carry
the same imposed bump as a common addend, and refitting the same series on
defensible sub-windows moves the decay constant itself by a factor of 2-3
(the "window range" column) while the covariance standard error claimed 2.9%.
Most samples in these windows do not individually clear their own leakage
floor. **What the lane supports is a BOUND: a damping difference larger than
roughly 1.5x is excluded** — the ringdown is a clean exponential across
several decades, so linearity holds over the relevant range. It does not
support "identical".

That bound is what makes the zonal result a source difference: equal damping
to within 1.5x, sustained level 2.7-2.9x, at two start states.

The lane's own limit: the excitation is basin-scale (its wall enrichment is
1.08), so it measures how each band damps a wave ARRIVING at it, not how it
damps something injected locally.

### Locus versus time — the statistic that separates a mode from a transient

Enrichment (share divided by area share; 1.0 = no localisation), day 180:

| | first sample | last half |
|---|---|---|
| legoESM meridional | **23.9** | 3.9 |
| legoESM zonal | 0.44 | **17.1** |
| NEMO meridional | 1.21 | 5.49 |
| NEMO zonal | 0.19 | 3.49 |

The two bands move in OPPOSITE directions. That is the whole finding: what
starts on the meridional walls decays away, and what ends up on the zonal
walls was not there at launch.

### Does it cancel in an ensemble difference?

Pre-registered, because a common-mode perturbation can still push
threshold-gated physics onto a different branch and the argument alone is not
sound. Two legoESM members from the same day-180 state, differing by the
ensembles' own 1e-14 relative perturbation (realised max relative dT 5.0e-14):

| member-minus-member enrichment | first sample | last half |
|---|---|---|
| meridional wall | **0.90** | 1.62 |
| zonal wall | **8.61** | 7.77 |

The meridional transient cancels (registered threshold was 3). **The zonal
band does not — it is the most enriched region of the difference field.**

### Is the cold turbulence start the owner? Pre-registered, NO

| day-180 bridged arm | wall first 8 | wall share, first sample |
|---|---|---|
| turbulence cold-started (production default) | 3.3456e-6 m | 0.852 |
| turbulence bridged from NEMO's restart | 3.3185e-6 m | 0.853 |

0.99x. Refuted. (The transfer gap is real and worth its own look; it just does
not make this transient.)

### The brief's own first candidate, closed for a different reason

The brief named "the non-halo-aware boundary stencil in one diagnostic". The
premise is wrong: it is the Hollingsworth-Kallberg kinetic-energy gradient
(`ke_gradient_scheme = "hollingsworth"`, NEMO `nn_dynkeg=1`), resolved ACTIVE
on the card under test, and it is a momentum-tendency term, not a diagnostic.
What closes it here is the other half: its boundary fill differs from the
oracle's convention only where a WET boundary row carries non-zero velocity,
and this run's own output has max |u| and |v| on dry cells of exactly 0.0 at
every step, so the two conventions coincide cell for cell. The missing halo
exchange is inert at single rank, which is how the twin runs. Both caveats
would bite on a wetting-drying case or a restart that does not zero land
velocities; that debt is recorded in `dino_wall_ldf_alignment.md`.

## The zonal-wall momentum budget — RUN, and it found a defect in legoESM's own instrumentation before it found anything about the walls

Pre-registered as RUN 3 before the probe existed, then re-registered as RUN 3b
when review found the first version was measuring an artifact. Nine
consecutive matched states (kt = 5760..5768): bridge NEMO's own state, before
level included, into legoESM; evaluate one per-term tendency breakdown;
compare against NEMO's own trend from the next dump. No time integration.

**Both adversarial reviews returned NO-SHIP on the first result, and both were
right.** What follows is the corrected version; the first version's null is
withdrawn.

### The defect worth more than the wall question: lateral friction was compared across time levels

NEMO evaluates lateral friction at the **before** level —
`CALL dyn_ldf( kstp, Nbb, Nnn, uu, vv, Nrhs )`, `stpmlf.F90:319` — because a
leap-frog-centred diffusion is unconditionally unstable. legoESM's production
path matches that. But `tendencies_with_diagnostics`, the supported per-term
breakdown every oracle budget in this campaign is built on, **silently dropped
the before-level argument its own sibling accepts**. So the probe compared
legoESM at *now* against NEMO at *before* for that term, and what it measured
was the difference between two time levels — a quantity the instrument
manufactured.

Prediction registered before the fix: the term's difference should fall by at
least 50x. **Measured: 1.59e-9 to 6.09e-12 m/s², a factor of 262.** Confirmed.

The fix is a two-line forward in the shared method, so it lands for every
caller at once. About twenty probes in this campaign call that wrapper and
none passed a before-level state — every lateral-friction term in every one of
those budgets on this card carries the same error. Re-running them is not this
branch's scope; the source is fixed and a test pins it.

### The reduction was cancelling the thing it was looking for

The first version reduced to a band **mean** per state and only then took the
2-step component. The sea-surface result does the opposite — per cell first,
then aggregate — and the order matters enormously: a band mean annihilates any
signal whose sign varies along the wall or down the column, which review
measured at 44x to 80,000x depending on term. A spatially signed source is
exactly what drives a free surface, because the surface responds to the
*divergence* of a difference rather than its mean. **The first null was a
property of my reduction, not of the model, and it is withdrawn.** The
statistic is now per-cell, matching the sea-surface measurement's own order.

### And then the corrected instrument disqualified most of its own channel

With the order fixed and a denominator added (the same statistic on the oracle
term, so a small difference can be told from "nothing there to find"), the
2-step comparison reads:

| term | legoESM / NEMO, zonal wall | same, meridional wall |
|---|---|---|
| lateral viscosity (u) | **1.0000** | **1.0000** |
| lateral viscosity (v) | **1.0000** | **1.0000** |
| KE + pressure gradient (u) | 2.0000 | 2.0009 |
| KE + pressure gradient (v) | 1.9999 | 2.0000 |
| vorticity + Coriolis (u) | 2.0000 | 2.0000 |
| vorticity + Coriolis (v) | 2.0000 | 2.0001 |
| vertical mom. advection (u) | 2.0222 | 2.0354 |
| vertical mom. advection (v) | 2.0114 | 2.0300 |

A factor of **exactly 2.000**, **identical on both wall bands**, on every term
whose time level is *not* verified matched — and **1.0000** on the two where it
is. That is a definitional offset between legoESM's start-of-step tendency and
NEMO's leap-frog-accumulated trend, not a wall phenomenon: a wall source would
not reproduce itself to four digits on a different wall. Comparing those
objects in the Nyquist band is comparing across a definitional boundary, which
is the "same transform on both sides" rule this campaign has been burned by
before.

**So the 2-step channel of this budget is valid for exactly one term.** For
lateral friction, time-level matched and reading 1.0000 against the oracle,
the lego-minus-NEMO difference carries **1.1e-5** of the oracle's own 2-step
content at the zonal wall. That is a genuine bounded null on that term. For
the other three terms no source statement can be made at all.

### What survives, and what it means

The **steady** channel is unaffected by any of this — both models' terms agree
there to between 1e-5 and 1e-2 relative — and vertical momentum advection
carries the largest steady zonal-wall enrichment. But that finding got weaker
under two checks:

* every term enriched in one velocity component is **depleted** in the other
  (vertical advection 3.85 on u against 0.50 on v; lateral viscosity the
  reverse), so neither reads as a term that is systematically worse at the
  wall;
* and a steady offset has **zero power at the Nyquist frequency by
  construction**, so it cannot excite a 2-step mode directly. Its only route
  is to push the wall's slow state away from the oracle's, so that the same
  broadband excitation gets a different response — which *is* the slow-field
  arm. Reported that way: a steady difference is evidence **for** slow-field,
  not a competing source story.

### What this budget cannot see — larger than the first version admitted

* **The whole barotropic split-explicit machinery.** Not the substep loop, not
  the filter weights, not the barotropic-before seeding, not the depth-mean
  reconciliation. That is where a 2-step free-surface mode lives.
  **But the surface-pressure-gradient term is the wrong place to go first:**
  it is *downstream* of the sea surface, so a 2-step surface mode produces a
  2-step pressure gradient by definition, on both sides. Measuring it would
  reproduce the ratio already in hand and name nobody. Symptom, not cause.
* **Three processes, not one, hide inside the vertical-mixing term.** The
  namelist sets implicit bottom drag, so drag and wind stress both enter the
  implicit solve and both read exactly zero as separate trends — on *both*
  sides (the probe now reports which side is zero rather than asserting it).
  Bottom drag is probably not the issue: these wall rows are 2231 m and 2558 m
  deep, so drag is confined to the bottom cell and remote from the surface
  mode. **Wind stress is a live candidate:** this card applies it as a
  post-step kick while NEMO deposits it in the implicit solve's top-cell
  right-hand side, and that deposit is not depth-mean-neutral. A placement
  difference, at a wind-driven wall row, with a direct path to the surface.
* A source that needs legoESM's **own** larger oscillation to exist. Every
  state is re-bridged from NEMO, so the trajectory never carries legoESM's
  mode forward and any amplification or feedback is invisible by construction.

### The band is not the two rows it was described as

Stamped now rather than asserted: the zonal band is rows **j=1 (48 cells,
69.5 deg S), j=14 (2), j=48 (2), and j=197 (48 cells, 69.5 deg N)** — the two
end walls plus two interior shelf edges. The earlier "rows j=1 and j=197" was
wrong, and the two end walls do not contribute equally.

### The self-check was vacuous, and is now proven not to be

The land control claimed the band mask was the only defence. It was not: the
3-D wet mask inside the volume weights zeroed the poison before the band mask
was ever consulted, so the control passed for a reason unrelated to its claim
— the **same defect class already found and fixed once in the sea-surface
probe in this same branch**. It matters here because 12.8% of the zonal band's
velocity cells are dry levels. The control now runs against genuinely unmasked
weights and the artifact records both arms: masked 0.7248 unchanged, unmasked
0.7248 to 671398.6. It can fail, and it is shown failing.

## The geography, PLAUSIBLE and no stronger

Row j=1 sits at about 69.5 deg S, the campaign's southern-basin
transport-deficit wall row, and vertical momentum advection is the term with
the largest steady difference there. The overlay against the basin torque-gap
row profile that would test a link was NOT run: with the two-step signature
absent, the link would be between the deficit and a STEADY term difference,
which is a different question from this branch's and belongs to the
southern-basin thread. Recorded as a lead, labelled PLAUSIBLE, not pursued —
"everything interesting happens at the boundary" is the null hypothesis a
correlation on a 100-cell band could not overturn.

## What was NOT done, and why

No fix and no A/B. The brief gated them on naming a source, the budget was run
rather than deferred, and it did not name one — it excluded a group and
pointed at the group it cannot see. Building an A/B now would mean choosing a
suspect by elimination among candidates that have already both been wrong.

**The named next step, and it is a re-projection of committed instruments
rather than new compute or new code.** The campaign already owns probes that
measure the per-step barotropic deposit difference at a bridged state, and one
that re-runs that measurement across a grid of states. Run that at NINE
CONSECUTIVE steps instead of ten widely-spaced ones and take the same per-cell
two-step projection used here. That reaches the machinery where the mode
actually lives, and it needs no new instrumentation.

Do **not** start at NEMO's surface-pressure-gradient trend: it is downstream of
the sea surface, so a two-step surface mode produces a two-step pressure
gradient on both sides by definition. It would reproduce the ratio already in
hand and name nobody.

Second candidate, cheap to test and currently mislabelled as unmeasurable: the
wind-stress placement difference described above.

The launch transient: stopped, deliberately.

## Instrument defects found and fixed, since the instrument decides what we believe

* the shared land poison plants a CONSTANT, which the Nyquist operator
  annihilates exactly — on this data it moved the statistic by a ratio of
  1.000000. An `alternating=True` flag was added to the SHARED helper.
* even alternating, it could not fail: the production area weights are already
  zero on land. The control now runs against UNMASKED weights, and a test
  shows a mask admitting land turns it red.
* the leakage floor was measured on one window and applied to another.
* the fit's identifiability gate was ANTI-correlated with identifiability —
  it accepted every degenerate fit and rejected the best one.
* the member-difference lane pointed both sides at the same file, so every
  ratio was 1.000 by construction. Equal paths are now refused without an
  explicit single-trajectory flag, which withholds the cross-model ratios.
* the registration diagnostic compared a response field against a full field
  outside the free lane (0.83 m against 1.8e-7 m). Nulled there.
* a check named "interior unchanged" was testing partition disjointness, not
  leakage. Renamed to say so.
* the measurement-scale plant control would abort at a recovery of ~1.73
  against a sign-coherent pre-existing wall mode — which is the hunted
  signature. It is now reported with a coherence flag instead of gated.
* the shared reduction keeps its per-sample loop: a vectorised sum agrees to
  twelve digits but not bit-for-bit, and a refactor must not move a published
  number. Verified bit-identical; a test pins it.

## Two stale comments in `dino.py`, reported not fixed

The base card's header says `nn_e=30` while the resolved value sixty lines
below is 23. And the forward-Euler block's "faithfully guarded OUT of this
frame (do NOT re-wire them here)" sits in the base dictionary that the
leapfrog card overrides — read literally it forbids what line 1573 does. That
is what sent me to the wrong line for retraction 1.

Also: neither the day-180 artifact nor its log records the recipe name or
whether the turbulence state was bridged, so the configuration behind those
numbers had to be inferred. Every arm on this branch records its own.
