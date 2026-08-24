# The southern-basin deficit: where it lives, and what it is not

Follow-on to `dino_verdict360_result.md`, which measured that one year of
legoESM from a shared NEMO restart matches NEMO in the circumpolar channel to
17 parts per million and is short by 0.95 Sv in the basin south of it, along a
path that is not monotonic. That verdict registered one next action: *a
term-by-term southern-basin transport budget*. This is that work.

Written for a reader who has not followed the campaign.

Pre-registrations, both written before their numbers existed:
`scripts/validate/ocean_fidelity/dino_1226/PREREG_basin_seasonal_decomp.md`
and `PREREG_basin_stage_budget_360.md`.
Probe: `basin_seasonal_decomp.py`. Series: `/tmp/dino_basin_seasonal_decomp.json`.

---

## The headline

1. **The deficit is a depth-uniform velocity error — a barotropic momentum
   problem, not a vertical-structure or water-mass one.** A single constant
   velocity difference of **−4.6e-4 m/s**, applied at every level, reproduces
   **76%** of the measured per-level difference by variance and **93%** of its
   total. legoESM's southern-basin flow is uniformly too weak in the eastward
   sense, by about half a millimetre per second, all the way down.
   **One qualification, added after Part 3.** "Uniform" describes the bulk of
   the column. In the bottom cell specifically — the level that sets the
   bottom-drag rate — the two models differ by **−1.5e-3 m/s** at day 360
   (−6.5e-3 against −5.0e-3), about three times the uniform value fitted on the
   same day. Any argument that runs through the bottom stress must use that
   number, not the uniform one.
2. **The water masses are not the cause.** The two models' southern-basin
   densities agree to 4e-5 kg/m³ on a 1027 kg/m³ field. Mechanisms that work by
   changing the water masses are ruled out at the basin scale, though the
   density *gradient* in the rows that own the gap has not been measured
   directly (see the caveat below).
3. **It lives against the southern wall.** The gap decays monotonically from
   the wall over about eight rows, and the four rows from 69.5°S to 68.4°S
   carry −0.905 Sv of the −0.952 Sv net. The shape is stable through the year
   (day-90 against day-360 row profiles correlate at 0.96; day-270 at 0.70).
4. **The surface forcing is identical, and that is measured.** The oracle's
   dumped wind stress matches the analytic form legoESM uses to 1e-10 per row,
   and the torque legoESM applies to these rows is 17.811 in every one of the
   36 ten-day windows of the year-long budget — DINO's wind has no time
   dependence at all. The whole difference is in the sinks and in how momentum
   is redistributed, not in what is put in. (The oracle's *vertical-solve* row
   is 18.100 rather than 17.811, but that row carries its bottom stress and
   vertical viscosity in the same number, so the 0.29 is not a wind
   difference.)
5. **The seasonal reading returns no verdict.** The registered test asked for a
   correlation of 0.7 between the gap and a forcing phase after removing its
   growth; the best measured is 0.48.

**A retraction, in place.** An earlier version of this document said the
deficit "grows with depth", with 82% of it below 500 m, and read the 92 m shift
of the transport's centre of mass as independent evidence of a
vertical-structure error. **That was wrong, and it was wrong in the way that
matters: it pointed the next work item at vertical mixing.** The per-level
transport difference grows with depth only because the model's layers grow from
10 m to 450 m thick; divided by each level's own wet cross-section, the
velocity difference is flat to three digits over the top 1000 m. And a
depth-uniform velocity difference *predicts* a centroid shift of −107 m against
the −92 m measured, so the centroid is a consequence of the uniform deficit,
not evidence against it. The probe now fits and scores both hypotheses side by
side.

---

## What DINO's forcing can and cannot do

The oracle's analytical surface forcing (`usrdef_sbc.F90`, `nn_forcingtype=4`)
builds the zonal wind stress from a fixed list of latitude/stress knots with no
time dependence at all, and the evaporation-minus-precipitation field is
switched off. **The wind does not have a seasonal cycle in this configuration**,
so no seasonal-wind explanation of the deficit is available. The only
*prescribed* seasonal channels are the solar flux (peaking 21 June) and the
target temperature the surface heat flux restores towards (peaking 21 July);
the salinity restoring is active but its target is time-constant.

Run day 0 of the verdict run is day-of-year 180 — 1 July, austral midwinter.
So day 90 is spring, day 180 midsummer, day 270 autumn, day 360 midwinter
again.

## The gap through the year

Southern-basin transport, legoESM minus NEMO, at the 19 days both models saved
a state. "floor" is how far apart four runs of the same model drift from a
1e-14 nudge, combined across the two models.

| run day | day of year | season | gap [Sv] | floor [Sv] | gap/floor |
|---:|---:|:--|---:|---:|---:|
| 10 | 190 | winter | +0.026 | 7e-10 | — |
| 90 | 270 | spring | −0.426 | 1.4e-04 | 2937 |
| 180 | 0 | summer | −0.232 | 2.2e-02 | 10 |
| 270 | 90 | autumn | −0.057 | 6.2e-02 | 0.9 |
| 330 | 150 | winter | −0.597 | 4.1e-02 | 15 |
| 360 | 180 | winter | −0.952 | 6.2e-02 | 15 |

Before about day 120 the four members have barely separated from their shared
restart, so the floor there is roundoff and the ratios are meaningless; read
them from day 150 on.

## The day-270 "agreement" is a cancellation, not agreement

Split the basin's transport into the part carried by the deepest wet cell times
the full water depth, and everything above it. The two halves are algebraic
complements, so this is one measurement and not two — but it shows the size of
what is cancelling. On the four-member mean rather than the single control
member:

| day | bottom-referenced part [Sv] | the rest [Sv] | net [Sv] |
|---:|---:|---:|---:|
| 90 | −0.62 | +0.21 | −0.43 |
| 270 | −1.74 | +1.56 | **−0.18** |
| 360 | −2.61 | +1.71 | −0.90 |

At day 270 the two models disagree by about 1.7 Sv one way and 1.6 Sv the
other; the net is small because those nearly cancel — a **19-fold**
cancellation. (On the single control member the same ratio is 53-fold and the
net is −0.06 Sv; the ensemble is the honest number.) Reporting day 270 as
"momentarily indistinguishable" is true of the metric and false of the physics.

This split is fragile by construction — it multiplies one cell per column by
3300 m — so everything below uses integrals instead.

## Where the transport sits in the vertical

Two hypotheses, fitted the same way to the per-level transport difference at
day 360 and scored on the same residual:

| null | explains, by variance | by total size | residual left |
|---|---:|---:|---:|
| a constant velocity difference of −4.6e-4 m/s at every level | **75.7%** | 65.5% | −0.070 of −0.935 Sv |
| legoESM = 0.938 × NEMO at every level | 37.5% | 52.1% | −0.336 of −0.935 Sv |

The uniform-velocity hypothesis wins on every score, and it also predicts the
observed shift in where the transport is centred (−107 m predicted, −92 m
measured) which the scaling hypothesis cannot produce at all. What is left over
after removing it is small and sits in the two deepest broad levels — 2890 m
and 3300 m — which is where an additional bottom-confined difference would go.

For orientation: both models spin this basin up hard over the year, from
5.9 Sv to 8.7–9.7 Sv, so these are two rapidly-adjusting trajectories, not two
climates.

## Where it lives in latitude

The basin south of the channel is 14 model rows, 69.9°S to 64.9°S. At day 360
the gap decays monotonically away from the southern wall — −0.277, −0.252,
−0.209, −0.167, −0.106, −0.076, −0.040, −0.005 Sv over rows 1 to 8 — and then
turns slightly positive, over a width of about 300 km.

* the four rows nearest the wall carry −0.905 Sv of the −0.952 Sv net, i.e.
  95% of the net but **80% of the negative signal** (the eight negative rows
  total −1.13 Sv and the six positive ones +0.18 Sv);
* eleven of the fourteen rows have a gap clearing twice their own noise floor;
* the shape is stable through the year: day-90 against day-360 profiles
  correlate at **+0.96**, day-180 at +0.89, day-270 at **+0.70**.

**Careful with the words "boundary layer".** The decay above is measured; the
label is a description. For the *rate* at which each model builds circulation —
a different quantity, measured in Part 3 — the answer is sharper and different:
both models deposit westward circulation in the six rows nearest the wall and
eastward beyond, and legoESM's westward lobe is **34% too strong** while its
eastward rows are right to **0.5%**. A rigid displacement of the profile
explains under 1% of the difference; two separate lobe gains explain 88%. For
the rate that is an **amplitude error, not a width error**, and it is the more
specific fact.

## Season: no verdict

After removing a straight-line growth, the gap's best correlation with any
forcing phase is **0.48** (against the 21-July restoring-temperature phase).
The pre-registration set 0.7 to confirm and 0.3 to refute, so the answer is
neither. Two further cautions, both registered in advance:

* the four forcing columns are two independent channels — the restoring target
  is an exact linear function of the 21-July cosine, and the solar flux nearly
  so;
* with one cycle and 19 correlated samples, a curve that is merely not straight
  will correlate about 0.5 with a cosine of that same single period by
  construction. There are roughly three or four independent samples here.

## Controls

Every one of these can fail, and all passed.

* The probe reproduces the parent verdict run's own day-90 and day-360
  southern-basin gaps **and** its member floors, to five decimal places.
* The three latitude bands sum to the full section on all 152 states read —
  checked on every state, including states served from the probe's cache.
* The per-level profile sums exactly to the transport it decomposes.
* The generalised bottom-referenced split reproduces the campaign's recorded
  channel-band split exactly, and the depth index it rests on is checked
  against the mask rather than assumed.
* legoESM's land mask equals NEMO's surface mask on all 10 348 cells.
* All four legoESM members stamp the same vertical ladder, the same seasonal
  clock and the same precision.

**A retraction.** The first version of this probe gated itself against a
day-90 gap of −0.4405 Sv taken from a code comment. That number belongs to a
different run. The gate failed on it and the constant was replaced with the
value the verdict run itself printed, −0.42578. A code comment is a pointer,
never a citable fact.

**A caveat on one instrument.** The density check above compares each model's
basin-mean density and, separately, the transport its density field predicts by
thermal wind. That second instrument reproduces neither model's own
bottom-referenced transport — it is short by 6–7 Sv on both sides, five times
the difference it is being used to exonerate — and the way it accumulates
across the basin makes it hundreds of times less sensitive to the interior rows
than to the end rows. Its verdict (that the density fields predict 1% of the
velocity difference) is therefore **plausible, not confirmed**. The basin-mean
density agreement is solid on its own; the row-by-row density gradient has not
been measured and should be, cheaply, from the states already saved.

## A note on weighting

The headline −0.952 Sv uses the model's reference layer thicknesses, the
metric the campaign has recorded throughout. The per-level, centroid and
bottom-referenced numbers use the actual (partial-cell) thicknesses and sum to
−0.936 Sv. The two weightings differ by about 3.5% within each model and that
difference very nearly cancels between them — the largest discrepancy over the
19 days is 0.028 Sv, below the noise floor at day 360.

---

# Part 2 — the accumulated momentum budget over the full year

Pre-registration: `PREREG_basin_stage_budget_360.md`, written before either run
started and disclosing in advance that this budget reduces to the
depth-integrated circulation and is therefore blind to how the transport is
distributed in the vertical.

## The two runs, and the control that makes them comparable

Both models were run again over the same 360 days from the same restart, with
every term of the momentum equation accumulated at every one of the 11 520
steps — legoESM inside its own step, NEMO through the instrumented build of the
oracle whose only namelist change is the run length.

**The instrumented oracle is bit-identical to the certified one over the whole
year.** Temperature, salinity and both velocity components agree to exactly
zero at day 90 and at day 360 against the verdict run. The accumulation is
therefore taken on the very trajectory whose gap is being explained, and the
instrumentation changes no physics at the new length.

On the legoESM side the same check was made through the circulation the budget
reduces to: a separate 10-day run reproduces the verdict member's day-10 state
to 4e-9 relative, which is the precision at which that member's velocity was
stored.

## The budget closes on the measured gap

| | legoESM | NEMO |
|---|---:|---:|
| rate from the accumulated per-step budget | +0.3750 | +0.4866 |
| rate measured from the states alone | +0.3749 | +0.4866 |

(band mean over the 13 wet southern rows, in cubic metres per second squared
per row; the year-mean rate at which each model builds eastward circulation in
this basin.)

The year-mean shortfall is **0.112**, i.e. legoESM builds southern-basin
circulation about **23% slower** than NEMO over the year as a whole.

**What this does and does not establish, said plainly.** Each model's own
budget reproducing its own state change to four decimal places is a real
*instrument* check — it proves the per-step term decomposition is complete on
both sides, that no term is missing or mis-signed, and that the accumulation
harness reads the same states the transport metric reads. But once each side
closes, the cross-model step (rate difference × time = state-gap difference) is
arithmetic, not physics. It is **not independent evidence** about the
mechanism, and the "0.3%" residual is simply the rounding of the quoted rate
difference; unrounded it is 0.05%.

**SUPERSEDED by Part 3, and the correction matters.** This paragraph said the
two models were in incompatible bases and named the collapse test as the next
measurement. The collapse is real — legoESM's realised rate equals its own
barotropic-solve row, and the two sides are in one basis — but it is an
**algebraic identity** on this configuration, not a measurement, and it
**narrowed nothing**: in a split-explicit scheme with this post-solve
correction, every term that can move depth-integrated momentum arrives through
that one row by construction. The sentence that followed, promising the test
would "exonerate every other term in one number", was wrong in the way that
wastes work — it described an identity as a discriminator. See Part 3.

## Per 10-day window: the mean is the only signal

Across the 36 windows the difference has mean +0.112 but a spread of 0.322 —
**three times its own mean** — and 13 of the 36 windows have the opposite sign.
The first quarter averages +0.199 and the last +0.406, but with nine windows a
quarter that separation is about 1.4 standard errors.

Consequences, both registered in advance:

* the pre-registered "roughly constant in time" prediction returned REFUTE by a
  hair (2.04× against a 2× refute bar) — but at this scatter that bar has almost
  no power: a series that really is constant at +0.112 would fail it more often
  than not. The registered test executed and returned REFUTE; the **question is
  open**, and the result must not be reported as measured growth;
* the day-260→270 window, which sits at the transport metric's near-zero
  moment, is **entirely unremarkable** here (+0.093 against a median of about
  +0.19), exactly as pre-registered. The mid-year "recovery" is a property of
  the transport metric's cancellation, not of the momentum deposit.

**No per-window or per-season attribution is supportable from this budget.**
Only the year mean is signal.

## Two structural facts worth carrying forward

**The forcing is identical, and it is measured rather than assumed.** The
oracle's own dumped stress matches the analytic form to 1e-10 per row; the
torque legoESM applies is 17.811 in every one of the 36 windows; and DINO's
wind stress has no time dependence at all. Whatever is different, it is not
what is being put in — it is the sinks and the redistribution.



**legoESM overshoots the wall's westward lobe.** NEMO's own year-mean deposit
into these rows is a meridional dipole: the five rows nearest the southern wall
receive *westward* circulation at −0.9 to −0.6, while rows further north
receive up to +2.0. The gap has the same dipole shape as NEMO's own deposit
(the two per-row profiles correlate at +0.94), with the node displaced about
two rows further from the wall. Read the signs directly: legoESM builds *more*
westward circulation than NEMO in exactly the rows where NEMO is already
building westward. This is not a failure to build the eastward flow — it is an
overshoot of the wall's return lobe, with the boundary layer too wide. That
points at the momentum sink and the momentum flux **at the wall**, and away
from anything interior or anything vertical.

## The stage-resolved half

Delivered in Part 3, from the completed artifact. The per-window numbers quoted
above were regenerated from that artifact through the committed instrument.

**A free reproducibility measurement fell out of that.** Two independent
360-day accumulations of the same configuration were run (the first aborted
before writing its artifact, so its numbers had been read off its printed
output). Comparing them: the oracle side is bit-identical, and the two legoESM
runs agree to **0.007** on any single ten-day window and to **0.004** on the
year mean — against a shortfall of 0.112 and a window-to-window spread of
0.322. So a single accumulation's year mean is good to about 4%, and the
window-level scatter reported above is physical variability rather than
run-to-run noise. The most likely source of the 0.007 is non-deterministic
accumulation order on the GPU amplified over 11 520 steps by the same chaos the
verdict run's ensemble measures; a third run would confirm it, and nothing in
this document turns on the difference.

**SUPERSEDED by Part 3.** The limitation recorded here — that legoESM keeps
part of the vertical solve's contribution to the depth-integrated circulation,
so the two stage tables are not row-comparable — was measured on an OLDER card.
The shipped card replicates the oracle's post-solve correction and the vertical
solve's net contribution is exactly zero on both sides. The caveat is withdrawn
— but withdrawing it buys no new power, because in the shared basis each side's
table has exactly one non-degenerate row.

---

# Ranked mechanism candidates — the ranking BEFORE the deciders ran

**This section is the record of what was ranked and why, before the two offline
deciders in Part 3 scored candidates 1 and 2. It is superseded by Part 3's
ranking and is kept because the deciders were registered against it.**

**None of these is confirmed.** This is the naming lane; no fix is built here.
Each is listed with the cheapest measurement that would settle it, cheapest
first. Every signature they have to reproduce is now specific: a **depth-uniform
eastward velocity deficit of about half a millimetre per second**, confined to a
**300 km boundary layer against the southern wall**, with **identical water
masses**, **identical wind input**, and a small extra residual in the two
deepest levels.

## 1. The momentum sink at the bottom — the barotropic drag

The oracle runs *non-linear* bottom drag, whose coefficient scales with the
local speed, and it applies it implicitly. That is a sink on the whole column,
not on the bottom cell alone, so it is the one candidate that naturally
produces a depth-uniform velocity error. It is also largest where the water is
shallowest — the wall rows — and it grows faster than linearly with the flow,
which fits a difference that grows through a spin-up year. The campaign has
already seen drag move this metric once: three drag fixes shifted the day-90
channel gap from 0.41 to 0.38 Sv.

**What is already ruled out inside this candidate**: the drag *law* and its
*constants* match. legoESM's shipped card resolves the non-linear NEMO drag
law, with the same drag coefficient (1e-3) and the same background kinetic
energy (2.5e-3 m²/s²) the oracle uses, with the drag on the implicit vertical
diagonal and a separate barotropic in-substep drag — the same three-part
composition the oracle has. The older linear-drag path and its background
velocity are not selected and are inert. So if drag is the owner, it is through
*where and how* the sink is applied — which velocity it sees, at which
substep, how the coefficient is averaged onto the faces — not through its
coefficients.

*Decider as registered*: compute each model's own per-row bottom-drag torque
from its bottom-cell velocities using the shared transcription, and compare.
**This was run (Part 3) and it does not settle the candidate either way** — it
constrains the shape, not the ownership. Retiring bottom drag needs a
perturbation arm.

## 2. The barotropic solve itself

A depth-uniform velocity error is, by definition, an error in the depth-mean
mode, and on the oracle side the depth-mean after a step is set **entirely** by
the split-explicit barotropic solve. Everything about that solve — its time
filter, the substep count, which velocity average it deposits, the face depths
it uses — is a candidate.

*Decider as registered*: apply to legoESM the collapse the oracle satisfies
exactly. **This was run (Part 3) and the sentence that followed it here was
wrong.** The collapse holds, but it is an identity and it exonerates nothing:
the barotropic-solve row is the depth mean of the entire explicit tendency plus
the wind, the substep drag and the barotropic biharmonic, so everything arrives
through it by construction.

## 3. The lateral momentum flux at the free-slip wall

The gap is a boundary layer, and the oracle's wall is free-slip. The width of
that layer is set by the lateral viscosity and by how the vorticity flux is
computed at the coastal boundary — a place the two models have historically
differed.

*Decider, offline*: measure the e-folding width of the per-row gap in each
model against that model's own frictional boundary-layer scale, and look at the
per-longitude structure of the difference. A broad, single-signed difference
points away from a coastal stencil; a few-column feature points straight at it.

## 4. The enhanced vertical viscosity in a convecting basin — **demoted**

This was the leading candidate before the vertical decomposition was done
correctly, on the reasoning that the basin convects all year and the two models
must decide the trigger on density differences far below any agreement two
equations of state can be held to. It is now fourth, for three independent
reasons:

* vertical mixing **redistributes** momentum within a column at fixed column
  mean; it cannot by itself create a depth-uniform velocity error;
* on the oracle side, the implicit vertical solve's contribution to the column
  mean is **discarded** before the next step, so a difference in it has no
  direct path into the quantity that is wrong;
* the oracle's own vertical deposit into these rows is 35 times larger at the
  far end of the basin than at the wall, while the gap is 8.6 times larger at
  the wall than at the far end — the two are anti-aligned.

It survives only as a possible *upstream* driver: a different near-bottom
velocity feeds the non-linear drag, which is a barotropic sink. It should be
run only if candidates 1 to 3 all come back null, and it is the only one on
this list that needs a fresh instrumented oracle run rather than an offline
measurement.

## 5. Close the density question properly

Not a mechanism, a loose end. The basin-mean density agreement is solid, but
the instrument used to convert that into "the density predicts 1% of the
velocity difference" is not validated and is insensitive to exactly the rows
that own the gap. Measure the row-by-row meridional density difference
directly, offline, from the states already saved.

---

# Part 3 — three offline deciders, and what they actually settle

Pre-registration: `PREREG_basin_deciders.md`, written before any number existed.
Probe: `southern_basin_deciders.py`. Everything here runs on artifacts already
on disk. **Sign convention, stated once: differences below are legoESM minus
NEMO unless the column says otherwise; the realised-rate deficit is quoted as
NEMO minus legoESM, +0.112 per row per year.**

**Read the summary first: none of the three deciders identifies the cause. Two
of them narrow the field, one of them turned out to be an identity, and the
per-term route is closed by an instrument gap. The honest state is that the
error is characterised precisely and its owner is not yet named.**

## Decider 0 — amplitude, or position?

Both models deposit *westward* circulation in the six rows nearest the southern
wall and eastward beyond, so the rate profile has a node. Two very different
errors look identical in a band mean.

Fit three shape models to all thirteen per-row differences:

| model fitted to legoESM − NEMO | explains |
|---|---:|
| a rigid meridional translation of the oracle's own profile | **0.8%** (best shift +0.06 rows) |
| one global gain | 29.9% (gain 1.081) |
| separate gains for the wall lobe and the rest | **87.8%** (gains **1.335** / **1.031**) |

**A rigid position error is refuted by its own residual.** legoESM's westward
wall lobe is **34% too strong** while the rest of the basin is right to 3%;
over the rows where the oracle's own profile is eastward the magnitude ratio is
**0.995**. This is the sharpest characterisation the campaign has of the
defect.

**A correction to how this was first argued.** The first version of this
section cited the node — legoESM crosses zero at row 6.44, the oracle at 6.24 —
and asserted "the node does not move". It does move, by 0.20 rows (4% of the
wall lobe's width), and more importantly the node cannot decide the question at
all: a width change confined to the interior of the wall lobe leaves it exactly
where it is, and a pure gain on one lobe cannot move it either. The node is
reported for completeness; the shape fits are the evidence.

## Decider 2 — an algebraic identity, reported as such

legoESM's realised circulation rate equals its own barotropic-solve row on
every row. At full precision the supporting cancellations are 4.2e-15 and
5.1e-14 against a scale of 1.7 — twelve orders inside the pass threshold, which
is the signature of a bin that could not have failed. Three exact facts force
it: two of the five stage rows are depth *deviations* and the reducer is a
depth *integral*; and the card's post-solve correction overwrites the
after-level column mean with the barotropic solve's own average.

**Two things in that table do have power, and they are the deliverable.**

* **The wiring.** The post-solve correction's row comes out at exactly minus
  the vertical-solve row (−0.388 against +0.388). With the correction disabled
  it would be zero and the realised rate would exceed the barotropic solve by
  that whole row. This is an arithmetic counterfactual read off the same table,
  not an A/B with the option switched off — strong, but call it what it is. It
  says the implicit vertical solve contributes **nothing net** to the
  depth-integrated circulation on either side.
* **The time filter is exonerated.** The per-step two-level rate and the
  endpoint-to-endpoint circulation drift do not telescope — the time filter is
  exactly what breaks the telescoping — and they agree to **0.02%** against a
  23% deficit.

**What it does not establish.** The barotropic-solve row is not the solver in
isolation: it is the depth mean of the *entire* explicit tendency (advection,
vorticity and Coriolis, pressure gradient, lateral friction) plus the wind
stress, the in-substep drag correction and the barotropic biharmonic. In a
split-explicit scheme with this post-solve correction, everything that can move
depth-integrated momentum arrives through that one row by construction. Nothing
was narrowed.

## Decider 1 — bottom drag: not shaped like the deficit, and not retired

The drag torque each model's own state implies, under one shared transcription,
differs by **+0.099** on a time-weighted year mean. The registered bin returns
**REFUTE**, and the reason is spatial:

| (both time-weighted) | four wall rows | rows 5–13 | ratio |
|---|---:|---:|---:|
| realised-rate deficit | +0.339 | +0.011 | **32×** |
| drag-torque difference | +0.125 | +0.112 | **1.1×** |

Row-wise correlation between the two shapes: **+0.36**. The deficit is sharply
wall-concentrated; the drag difference is flat across the basin. The
pre-registered concentration clause is not met.

**Three retractions, all logged in the probe where they happened.**

1. The pre-registration read a positive difference as "legoESM's sink is
   weaker", assuming the bottom flow here is eastward. It is westward in both
   models, so a drag sink there pushes *eastward* and a positive difference
   means legoESM's drag pushes *harder*.
2. The replacement sentence — "drag opposes the anomaly, therefore it is a
   consequence not a cause" — is **also wrong**, and this one is a logical
   error. For *any* sink-type cause the same pattern appears: too strong a sink
   gives a weaker equilibrium flow, and evaluating the shared law on the two
   states then yields a difference that opposes the anomaly. The sign carries
   no information, and it is no longer part of the verdict.
3. An earlier draft reported +0.116 and called it "103% of the shortfall", from
   a plain mean over 19 non-uniformly spaced days, ten of which sit in the last
   quarter. Time-weighted it is +0.099 — and the ratio to the shortfall is no
   longer quoted at all, because this diagnostic is the full-velocity drag
   torque, whose column mean the post-solve correction discards every step. It
   is not a term in the budget the deficit lives in, so the ratio compares two
   different budgets.

**One more limit.** At these speeds the oracle's background kinetic energy
keeps the drag rate within 0.5% of constant, so the "non-linear" law is running
in its linear regime and this measurement is close to a restatement of the
velocity difference itself.

**Bottom drag is therefore not retired.** What the model applies is a substep
integral of an evolving depth-mean flow, which no offline diagnostic can see.
Only a perturbation arm can close it.

## Decider 3 — the per-term table closes, and still cannot attribute

With both models' per-term momentum trends accumulated over the same 360 days
and reduced identically, the four term groups **close to 2e-9 per row** against
each model's own realised rate.

**That closure is a correction to this document's previous version, and the
correction matters.** The first version of this section reported a closure
*failure* of 49 per row and published it as "the two models' term sets are not
in bijection at this level", then commissioned "building a matched term
correspondence" as the next piece of work. **That was a bug in the probe, not a
property of the models**: the oracle's barotropic operand was being credited
into the surface group while its post-solve row was left out, double-counting a
number of order 1e4. Both reviewers found it independently and both verified
the repricing. The recommended work item it generated is withdrawn.

**But the closed table does not attribute anything.** In the wall rows the four
groups differ by 105 in total magnitude and sum to 0.339 — a **310:1**
cancellation. Reading a 0.3% residual out of groups that individually differ by
4–8% is not a measurement. And the pattern of those differences — the totals
agreeing to 0.02% while every individual term is off by several percent — is
the signature of a time-level or staging difference in the *diagnostics*, not
of physics. Nothing may be drawn from it until the group differences are shown
to survive a time-level control.

Two structural notes, both of which limit the table further: legoESM's last
group is defined as a remainder, so its side closes by construction and only
the oracle's side could ever have failed; and legoESM's dominant
wall-concentrated group is 99.7% a single fused diagnostic (kinetic-energy
gradient plus pressure gradient) that the model cannot currently split.

## Where this leaves the candidates

The signature every candidate must now reproduce: **legoESM's westward
circulation lobe against the southern wall is 34% too strong, the rest of the
basin is right to 3%, a rigid displacement of the profile explains under 1% of
the difference, the water masses are identical, and the wind input is identical
to 1e-10.**

1. **Whatever sets the sea-surface set-up against the wall.** A boundary
   current 34% too strong in near-geostrophic balance implies a cross-shore
   surface slope about 34% too large. **This is the top work item and it is
   fully offline**: both models already save the sea-surface field, so
   comparing the cross-shore slope in the four wall rows is minutes of work on
   existing files. It is a *balance* test rather than another bookkeeping
   decomposition, which is exactly why it is immune to the 310:1 cancellation
   that makes the term table useless. If legoESM's slope is high by roughly the
   same factor as its velocity, the lobe is geostrophic and the owner is the
   barotropic solve's treatment of the wall. If the slope matches while the
   velocity does not, the lobe is ageostrophic and the owner is friction or
   advection at the wall.
2. **The lateral momentum flux at the free-slip wall** — the ageostrophic arm
   of the same test, promoted from third.
3. **The barotropic mode's Coriolis term inside the substep loop**, where the
   meridional-velocity stencil at the wall-adjacent row is one-sided. Demoted
   from a previous draft's top pick, for two reasons: the vorticity and
   Coriolis group's wall difference is 4.8× smaller than the pressure and
   kinetic-energy pair's, and a one-sided stencil is a one- or two-row defect
   while the measured error decays smoothly over eight rows and leaves the
   eastward lobe right to 0.5%. That is the signature of something scaling a
   whole boundary current, not of something breaking one row.
4. **Bottom drag** — not shaped like the deficit, not retired. One arm with a
   halved drag coefficient over 90 days would close it; the wall rows must move
   from 1.34 toward 1.0 by the predicted amount.
5. **The enhanced vertical viscosity** — dead as a *direct* contributor (its
   entire column-mean effect is removed on both sides), alive only as an
   upstream path, because the bottom-cell velocity difference is larger than
   the depth-uniform one and that is what any drag term sees.

**Positively closed, with numbers.** The wind stress matches to 1e-10 per row
and is constant across all 36 windows. The face-thickness convention (minimum
versus arithmetic averaging) accounts for about 1% and is *anti-aligned* with
the deficit — refuted as owner, logged as convention debt. The time filter is
exonerated at 0.02%.

**What none of this is.** No perturbation test has been run. Nothing above may
be called confirmed until one arm changes one thing and moves the wall rows in
the predicted direction. Three of this lane's own conclusions were retracted
after review — a claimed depth dependence that was a layer-thickness artifact,
a sign argument that was a logical error, and a closure failure that was a
pairing bug — so the standing instruction for the next lane is to run the
balance test above before building anything.


---

# Part 4 — the balance test: the lobe is geostrophic, and the owner is probably friction

Pre-registration: `PREREG_wall_balance.md`, written before any number existed.
Probe: `southern_wall_balance.py`. Read-only, from states already on disk.

## Why this test and not another decomposition

Every bookkeeping route into this defect is closed by cancellation: the stage
table is an identity, and the per-term table closes to 2e-9 but its groups
cancel 310:1 against what they would have to explain. A balance test asks
whether the flow is in the balance it should be in, and the quantity of
interest is its numerator, so cancellation cannot swamp it.

## The result: the wall lobe is in geostrophic balance

Decomposing the circulation difference in the four wall rows at day 360:

| term | Δ [10⁶ m³/s] | share |
|---|---:|---:|
| **circulation difference (legoESM − NEMO)** | **−10.54** | 100% |
| of which the sea-surface slope | −9.59 | **91.0%** |
| of which the density gradient | −0.71 | 6.7% |
| **ageostrophic residual** | **−0.24** | **2.3%** |

Robustness, all measured: the surface-slope share alone reads 94.1% at day 360,
90.3% as the ratio of the 19-day sums, 88.7% time-weighted and 78.9% as the
mean of daily ratios — every one clears the pre-registered 70% bar. Across the
four ensemble member pairs it is 93.3–95.6%, a spread of 2.4 points. The
circulation difference is 19 ensemble floors wide and the slope difference 22.

**Registered verdict: CONFIRM geostrophic.**

**The density term was measured, not assumed.** The pre-registration dropped it
because the two models' *basin-mean* densities agree to 4e-5 kg/m³ — an
assertion about a mean, not about the meridional gradient in four rows. Measured
with a common reference profile (so the huge hydrostatic common mode cancels
rather than being differenced), it is 6.7% and *same-signed*, so it adds to the
explanation. Three independent implementations — this probe's and both
reviewers' — agree at 6.7–6.9%.

## But the registered *consequence* does not follow, and the width says why

The pre-registration said a confirm would name the barotropic solve's wall
treatment as the owner. **It does not.** Geostrophy is a diagnostic relation:
*any* wall-trapped momentum error — friction, advection, the boundary vorticity
— adjusts to it within days and would produce exactly this table. What the test
establishes is a **target** (the wall's sea-surface set-up carries the
difference) and an **elimination** (the lobe is not ageostrophic, so work aimed
at an ageostrophic wall imbalance would have been misdirected).

The **length scale** does discriminate, and it points the other way. legoESM
piles about **4.4 mm** too much sea surface against the wall, decaying with an
e-folding scale of **2.6 rows ≈ 112 km**, with no basin-wide offset (0.06 mm).
A barotropic-solver wall artifact lives at the grid scale (1 row) or at the
barotropic deformation radius (~1500 km here). Neither is 112 km. The
frictional (Munk) boundary layer is: with the oracle's own viscosity at these
rows — 5387 m²/s, from its namelist formula on its own mesh — the Munk width is
**87 km = 2.0 rows**.

**So the leading hypothesis is now the lateral-friction boundary layer at the
wall, not the free-surface solve.** One coincidence, one number: **PLAUSIBLE**.

legoESM sets the same viscosity coefficient by construction — its card's
viscous velocity scale is 0.27 m/s, the oracle's `rn_Uv`, embedded as
½·U·max(e1,e2) inside the same divergence–curl operator — and the probe's own
gate pins the two models' grid spacings to 1.3e-5. So a coefficient mismatch is
not the explanation; if friction owns this, it is in how the operator meets the
free-slip wall, not in its magnitude.

## A land value inverted this verdict once

Row 0 is entirely dry, and both models store a sea surface of exactly 0.0 there
while the wet surface nearby is near −0.97 m. A centred meridional difference at
row 1 therefore straddles a one-metre step that is a land value. Before that
stencil was masked, row 1 returned a **wrong-sign** contribution large enough to
drag the wall mean from 94% to 37% and the verdict from *confirm* to *"leaning
refute — the owner is friction or advection at the wall"*. **The branch was
inverted by a land value.** Row 1 is now recovered with a one-sided difference
that touches no land, and a planted value on the dry row is required to move no
scored row.

## Pre-registration clauses that did not survive contact

* **The slope-ratio clause is withdrawn as mis-specified.** The slope-only
  circulation deliberately omits the density term, so its ratio could never
  approach the target however geostrophic the lobe was.
* **The window for the fraction was not named.** All four aggregates are
  reported; the ratio of the sums is the defensible one and every reading
  clears the bar, so the choice decides nothing.
* **The consequence clause is reported as not following**, per the argument
  above, rather than quoted.

## Two controls that came with it

**The per-term group differences are physical, not diagnostic staging — and the
parent document's claim to the contrary is retracted.** Part 3 said those
differences "have the signature of a time-level or staging difference in the
diagnostics rather than of physics". Measured in the first ten days, when the
two trajectories still agree to storage precision, the magnitude-weighted ratio
of window-1 to year-mean differences is **0.072** — they grow with the
separation, so they are physical. Reported as **plausible** rather than
confirmed: a staging error would scale with the terms themselves, and the terms
are also smaller early, which narrows the margin over the refute bar.

**legoESM's fused pressure-gradient diagnostic carries the hydrostatic gradient
only.** The deciding fact is in the routine that builds the pressure anomaly: it
iterates the equation of state against the *reference* thickness precisely to
avoid double-counting the surface-gradient forcing the barotropic solver
applies. The per-term pairing was therefore correct and the group differences
are not a pairing artifact.

## What the next lane should do first

1. **The free-slip wall condition in the lateral-viscosity operator** — how
   legoESM's divergence–curl form meets the boundary, against the oracle's
   `rn_shlat = 0`. The coefficients match; the boundary treatment has not been
   compared.
2. **A one-variable viscosity ablation.** Munk scaling predicts the wall-row
   excess moves as the cube root of the coefficient, i.e. −26% for a doubling.
   Moves as predicted → the owner is named. Barely moves → friction is
   exonerated and the barotropic solve returns to the top.
3. **The barotropic face depth at the last wet row**, offline, both models.


---

# The post-tendency stages, bisected

Follow-on, 2026-08-23. Everything above compared terms *inside* the explicit
momentum tendency. This section covers the stages that run *after* it, which had
never been compared against the oracle at all.

Pre-registration (written before any number existed):
`scripts/validate/ocean_fidelity/dino_1226/PREREG_post_tendency_stage_birth.md`.
Probe: `post_tendency_stage_birth.py`. Lane: day 180, one step, kt 5760 → 5761,
where the two models are identical at entry — so every number below is a pure
operator difference with no trajectory feedback in it.

Written for a reader who has not followed the campaign. Every difference is
legoESM minus the oracle, and the deficit this campaign is chasing is **negative**
in that convention.

## The table

Each row is the difference that exists *at* that point in the step; the second
table is the difference *born* between one point and the next. Everything is
thickness-weighted over the four rows against the southern wall.

| point in the step | wall velocity difference [m/s] | wall transport difference [Sv] | how wall-concentrated |
|---|---|---|---|
| after the barotropic solve and the leap-frog recombination | −2.00e-6 | −0.0468 | 0.11× |
| after the implicit vertical solve | −2.00e-6 | −0.0467 | 0.11× |
| after the after-level reconciliation | +2.25e-8 | +0.00049 | 0.34× |
| the committed state | +2.25e-8 | +0.00049 | 0.34× |

| stage | difference BORN in it [m/s] | [Sv] | clears its registered bar? |
|---|---|---|---|
| barotropic solve + leap-frog recombination | −2.00e-6 | −0.0468 | yes, 12× and 468× over |
| implicit vertical solve | +4.7e-10 | +1.05e-5 | **no** — 340× and 9.5× under |
| after-level reconciliation | +2.03e-6 | +0.0472 | yes, 13× and 472× over |
| time filter and commit | 0 | 0 | no |

## What it says

1. **The two big stages are one object, and it nearly cancels.** The barotropic
   solve deposits a large depth-uniform difference and the after-level
   reconciliation removes 99 % of it — which is what that stage is *for*: it
   overwrites the column mean with the barotropic solve's own answer. Reporting
   either of them alone as a "birth" would be the mistake this campaign has
   already made once, of budgeting across a boundary where the state changes
   representation. **Read them as a pair.**
2. **Netted over one whole step, nothing clears the bar.** The committed
   difference is **+2.2e-8 m/s**, seven times *under* the per-step bar, and it
   has the **wrong sign**: the deficit is negative and this is positive. A stage
   that pushes the other way cannot own it however large its internals are.
3. **Nothing is wall-shaped.** Every point in the step scores 0.11–0.34 on the
   wall-concentration measure, against a registered bar of 2.0. The difference
   these stages make is basin-wide, and the deficit is not.
4. **The implicit vertical solve is the smallest term in the step**, 340 times
   under the velocity bar. It is not the owner.

**One honest caveat about the bars, and it changes how row 2 reads.** The two
magnitude bars were derived from two different published figures — the deficit's
depth-uniform velocity (4.6e-4 m/s) and the wall band's transport gap
(0.288 Sv) — and those two describe different regions, so they are not mutually
consistent: they disagree by about a factor of thirty on this band's
cross-section. The committed one-step difference of +2.2e-8 m/s is seven times
*under* the velocity bar but +4.9e-4 Sv is five times *over* the transport one.
The refutation therefore does **not** rest on magnitude. It rests on the two
legs that are unambiguous and that the pre-registration fixed in advance: the
committed difference has the **wrong sign** (positive, where the deficit is
negative), and the **wrong shape** (0.34 on the wall-concentration measure
against a bar of 2.0, i.e. the difference is nine times larger in the interior
than at the wall). A birth pushing the other way, spread over the wrong rows, is
not the owner however large it is.

**Verdict, CONFIRMED for the day-180 state:** none of the four post-tendency
stages is born with the right sign or the right shape to be the southern
deficit's owner at this state. As registered, this refutes them
under *linear* retention only — this campaign has already measured a badly
non-linear map from operator error to transport, so the honest statement is that
the deficit is not injected here, not that these stages can never matter.

## The implicit solve's "15 %", decomposed

The one end-to-end number ever attached to the implicit vertical solve was a
"15 % discrepancy", carried in prose as the campaign's largest unexplained
figure. It does have a probe behind it and it reproduces exactly. It is a
**root-mean-square amplitude ratio** — legoESM's after-level velocity is 15.3 %
larger in RMS than the oracle's (23 % for the north-south component) — measured
**end-to-end**, on the year-5 state, with every upstream difference inherited.
It is not a per-point error and it is not the solve's own.

Feeding the solve the oracle's *own* input and comparing against the oracle's
own output separates it, on the day-180 state:

| | u | v |
|---|---|---|
| the solve's own operation, oracle input in | **2.2 %** | **1.4 %** |
| the input the solve is handed, like-for-like | **0.66 %** | — |
| end-to-end, as the 15 % figure was measured | 30 % | 38 % |
| doing nothing at all (the no-op reference) | 11 % | 4.9 % |

And the end-to-end figure splits cleanly: the *baroclinic* part of legoESM's
post-solve velocity matches the oracle's to **2.2 %**, the same as the seeded
arm. All of the remainder lives in the depth-uniform part, where the oracle's
dump has had the barotropic mode removed and legoESM's has not — the oracle
takes it out before the solve and puts it back two stages later, legoESM strips
and restores it inside the stage.

**So the 15 % is a representation mismatch at a stage boundary, not a physics
error, and the solve's own operation is a 1–2 % effect.** The campaign's largest
unexplained figure is explained, and the "gate row is blank" framing that went
with it is withdrawn: the row was never blank, it carried this number plus a
note saying it did not isolate the solve.
