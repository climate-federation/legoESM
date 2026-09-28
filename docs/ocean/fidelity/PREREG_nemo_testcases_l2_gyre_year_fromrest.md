# PREREG — the GYRE from-rest YEAR and its spread floor

> **OUTCOME (added after the fact; the preregistration itself is unchanged
> below).**  PHASE 0 is **BLOCKED**.  legoESM's certified GYRE card aborts at
> **step 48 of 2160** because its prognostic TKE closure diverges
> super-exponentially from about step 38, while every dynamical field stays
> healthy.  Every expectation P1-P5 is therefore **UNMEASURED** -- neither held
> nor refuted.  See
> `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_year_fromrest_receipt.md`.

Written BEFORE any member runs, legoESM or NEMO.  It is the GYRE counterpart of
the DINO from-rest protocol
(`scripts/validate/ocean_fidelity/dino_1226/PREREG_verdict360_fromrest.md`, on
branch `fidelity/dino-*`), and it is a DIFFERENT measurement from the kt=1..10
bit ladder that rounds 8..47 of this branch run.

Harness: `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_year_fromrest.py`.
NEMO acquisition: `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_year_fromrest_members/run.sh`.
Artifacts: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest/`.

## 0.  The question

NEMO 5.0.2 and legoESM, run **completely independently from rest** on the
certified GYRE card, must give the same answers — at least to within each
model's own run-to-run spread.  On GYRE that has never been measured beyond
**ten time steps** (40 hours).  This round measures it over a **year**.

The ladder's current state, at the tip of this branch, is the entering
condition:

| kt=2 row | normalized max | status |
|---|---:|---|
| `GYRE-zco.kt2.before.T` | `5.786374e-16` | AT-BAR |
| `GYRE-zco.kt2.before.S` | at bar | AT-BAR |
| `GYRE-zco.kt2.before.u` | `2.7478406377547115e-12` | DEBT |
| `GYRE-zco.kt2.before.v` | `3.305560306813421e-12` | DEBT |

(receipt `nemo_testcases_l2_gyre_phase3_round8_receipt.md`, the re-pinned
sweep; `u`/`v` normalise by `max(|oracle|,1)` and at kt=2 that denominator is
1, so those two rows are ~3e-12 **m/s absolute**.)

So at kt=1 the two models are bit-identical and at rest; the whole difference
is manufactured by the operators.

**It does not stay at 3e-12.**  The same receipt records the ladder's kt=10
row at T `5.636638587568748e-3` normalized, and a *pre-measurement taken while
writing this document* — legoESM's own ten steps against NEMO's kt=10 restart,
from data already on disk, no new run — gives, on NEMO's `tmask`, interior
32x22x30, fp64:

| row | after 10 steps (40 hours) |
|---|---:|
| `T3D` rms | `2.768186e-03` K (max `1.340785e-01` K, field max 23.4954) |
| `S3D` rms | `1.731378e-04` g/kg (max `5.876136e-03`) |
| `SSH` rms | `5.427653e-06` m (max `3.830982e-05`) |

`0.134785 / 23.4954 = 5.737e-3`, one step past the receipt's `5.6366e-3`, so
the two readings converge and the ten-step number is not a new instrument's
first output.

That pre-measurement also **pins the restart's time level** (Rule 1d), which
this round's whole comparator depends on.  `restart.F90:176-182` writes
`'tn' <- ts(:,:,:,jp_tem,Kbb)` in the RK3 branch, and the swap `Nbb <- Naa`
happens before `rst_write` in the step routine THIS CARD COMPILES --
`cfgs/GYRE_OMIP_L2_P3_SM_R41ADVSP/MY_SRC/stprk3.F90:237` and `:280`.  (The
shipped `src/OCE/stprk3.F90` is the same code at `:213` and `:256`; a review
read the shipped file and flagged this citation as off by 43 lines, which is
exactly the offset between the two.  The card runs its MY_SRC copy.)  So the
restart stamped `kt=n` holds the state **after n completed steps**, which is
legoESM after n `model.step` calls.  The CODE READING is the evidence; the
numerical agreement above only corroborates it, and weakly -- one step's
tendency at kt=10 is ~1e-3 K against a 2.8e-3 K gap, a factor of about two, so
the alternative mapping would NOT have been unmistakable.  An earlier draft
called it unmistakable; that overstated it, and a review said so.

**So GYRE's entering condition is nothing like DINO's.**  After FORTY HOURS the
two models already differ by 2.8e-3 K rms — the size of DINO's gap after a
YEAR.  Whatever this round finds, it is not a question of chaos slowly
overtaking a tiny offset.

## 1.  What DINO measured, so GYRE's numbers land in context

DINO, same protocol, 3-D wet-cell temperature rms [K]
(`/data/abyssal/dbalwada/dino_fromrest_y1/verdict360_fromrest/phase0_floor.json`):

| day | 30 | 90 | 180 | 360 |
|---|---:|---:|---:|---:|
| floor (legoESM 4-member) | `1.245e-5` | `8.811e-4` | `1.383e-3` | `5.450e-3` |
| gap (legoESM vs NEMO) | `2.039e-3` | `2.304e-3` | `4.590e-3` | `3.924e-3` |

verdict `gap/(2*floor) = 0.360` at day 360 → **INDISTINGUISHABLE**, and Phase 1
(the NEMO members) was never spent because even a zero NEMO spread could not
flip it (`phase1_worth_running: false`).  A later re-measurement of the same
gap on the fixed card gives day-30 `6.889e-4 K`
(`/data/abyssal/dbalwada/dino_fromrest_y1/day_gap_after.json`); the table above
is the pinned artifact and is what this document compares against.

DINO's shape: the **floor grows 438x between day 30 and day 360** while the gap
grows 1.9x.  Chaos overtook a nearly-flat offset.

## 2.  The runs

Both models: **from rest, 360 days, 2160 steps at `rn_Dt = 14400 s`**, snapshot
every 30 days (180 steps): days 30, 60, … 360.

Four runs per model: a **control** (seed 0, no perturbation) and **three
members** (seeds 1, 2, 3).

### Why 2160 steps, and a claim this document has already RETRACTED

`nn_leapy = 30` (certified `namelist_cfg`, echoed at
`gyre_kt1_10/ocean.output:233`) selects 12 x 30-day months, so
`nyear_len(1) = 360` and one year is exactly `360 * 86400 / 14400 = 2160`
steps.  `ndate0` appears **nowhere** in the certified `namelist_cfg`; the run's
own output file is named `GYRE_OMIP_L2_P3_40h_00010101_00010102_gr_0000.nc`,
so the start date resolved to 0001-01-01 -- read off the run, not inferred.

**RETRACTED, before any member ran.**  An earlier draft of this section claimed
2160 was the ONLY defensible length, because GYRE's forcing subtracts
`(nyear-1)*rjjhh*zyydd` (`usrdef_sbc.F90:89-91,160-163`) while legoESM's
transcription (`nemo_recipe.py:708-735`) has no `nyear` term, so the two could
only agree inside year 1.  **That is false.**  Every cosine in the routine has
denominator 4320 h (`usrdef_sbc.F90:94,96,164`), i.e. period
`2*4320 = 8640 h = 360 days`, which is exactly what the `nyear` term subtracts
per year.  The subtraction is an integer number of periods and therefore a
**no-op at every kt**; one reviewer evaluated the difference at kt = 2161,
2400, 3000 and 4320 and got 1.1e-16 in all three cosines.  legoESM's
transcription is exact at any run length.

So 2160 steps is a **CHOICE**, and it is in the ASKED table as one.  It is
chosen because it is exactly one seasonal cycle and because it puts GYRE's
scored days on DINO's, which is the comparison this round exists to make.  Note
also that the card itself declares `n_steps = 4320`
(`nemo_testcase_recipe.py:851`), a two-year target that no run in this campaign
has used; this round does not adopt it.

**What day 360 is, seasonally.**  Day 360 of a 360-day year starting 1 January
is 26 December -- five days past `ztimemin1`, the seasonal MINIMUM of the
forcing and the maximum of northern convection.  Day 180 is 27 June, near
`ztimemax1`, peak stratification.  The two are therefore the seasonal
EXTREMES, not two samples of one regime, and both are reported as headline
days rather than day 360 alone.

### The NEMO namelist rows that change, and only these

Base: `cfgs/GYRE_OMIP_L2_P3_SM_R41ADVSP/EXP00/namelist_cfg` (the certified
phase-3 card, 10-step ladder).

| row | certified ladder | this round | why |
|---|---:|---:|---|
| `nn_itend` | `10` | `2160` | one year at `rn_Dt = 14400` under `nn_leapy = 30` |
| `nn_stock` | `10` | `180` | a restart every 30 days = every scored day |
| `nn_write` | `10` | `2160` | one output file instead of 216; scored quantities come from restarts, so this is measurement-neutral |
| `nn_pert_seed` | absent | `0` / `1` / `2` / `3` | the ensemble member; **`0` is defined to be the unperturbed path** |

`rn_Dt`, `jpkglo`, `nn_GYRE`, every `&namtra_*`, `&namdyn_*`, `&namzdf*`,
`&nameos`, `&namdrg`, `&namlbc`, `&namsbc*` row is **unchanged**.  The run.sh
diffs the member namelist against the certified one and prints every differing
row; more than the four rows above is a refusal.

## 3.  The perturbation, and why GYRE needs a source patch where DINO did not

DINO already carries `nn_pert_seed` (`cfgs/DINO/MY_SRC/usrdef_istate.F90:177-183`):

```fortran
IF( nn_pert_seed /= 0 ) THEN
   DO jk = 1, jpk
      pts(:,:,jk,jp_tem) = pts(:,:,jk,jp_tem) + 1.e-10_wp                    &
         * SIN( REAL( NINT(pdept(:,:,jk))*73 + NINT(gphit(:,:)*1000._wp)*179 &
                    + nn_pert_seed*997, wp ) ) * ptmask(:,:,jk)
   END DO
ENDIF
```

GYRE's `src/OCE/USR/usrdef_istate.F90:37-79` has **no such block and no
`gphit`**, and `src/OCE/USR/usrdef_nam.F90:64` has no `nn_pert_seed` in
`namusr_def`.  So this round ships an additive MY_SRC patch that copies the
DINO statement verbatim into GYRE's `usr_def_istate`, plus the declaration and
the namelist entry.  The patch:

* adds `USE dom_oce, ONLY: gphit` and `USE usrdef_nam, ONLY: nn_pert_seed` to
  `usrdef_istate`;
* adds the `IF( nn_pert_seed /= 0 )` block **after** the existing `DO_3D`
  profile loop, so the unperturbed arithmetic is untouched;
* adds `INTEGER, PUBLIC :: nn_pert_seed = 0` to `usrdef_nam`;
* extends the single line `NAMELIST/namusr_def/ nn_GYRE, ln_bench, jpkglo`
  (`usrdef_nam.F90:64`).

That last item is the **one** line the patch does not merely add.  Fortran has
no way to extend a `NAMELIST` statement without rewriting it, so the run.sh
enforces the weaker mechanical property that actually matters: **exactly one
removed line, and it must be that `NAMELIST/namusr_def/` statement.**

**The instrument's own check.**  The claim "seed 0 is the unperturbed path" is
not argued, it is measured: the control member's day-360 restart and its
`ocean.output` must be **byte-identical** to an unperturbed run of the same
namelist on the pristine binary.  The run.sh performs that comparison by
running the certified `R41ADVSP` binary's card at `nn_itend = 2160` alongside
the patched one, and refuses on any difference.

Two properties of the perturbation are asserted by the harness rather than
assumed:

1. it is **1e-10 K ABSOLUTE**, not relative — about 5e5 ulp on a ~20 K field,
   far above fp64 rounding and physically meaningless;
2. **it is NOT zonally uniform on GYRE, unlike on DINO.**  GYRE's grid is
   rotated 45 degrees (`usrdef_hgr.F90:125-126`:
   `pphit = zphi0 - zim05*ze1deg*zsin_alpha + zjm05*ze1deg*zcos_alpha`), so
   `gphit` varies along a row by 20.9 degrees on the card.  Copying DINO's
   zonal-uniformity assertion would fire immediately and would be measuring
   nothing.  The harness instead asserts the property the expression really
   has: at fixed level, the perturbation is **constant within each set of cells
   sharing `NINT(gphit*1000)`** (zero variance within group, exactly), and it
   is checked to be non-degenerate (more than one group, and more than one
   distinct value).

legoESM reproduces the statement in the harness, including Fortran `NINT`'s
round-half-**away**-from-zero (`np.round` is half-to-even), and using NEMO's
OWN operands.  `pdept` at the call site is **`gdept(:,:,jk,Kbb)`**
(`istate.F90:127-130`), not `gdept_0`; under `key_qco` those are equal only
because `ssh = 0` at initialisation, which this card's
`usr_def_istate_ssh` sets (`usrdef_istate.F90:82-103`).  The dependence is
stated because it is a condition, not an identity.  `gphit` and the depth are
read from `mesh_mask.nc` of the certified run and are required to agree with the card's
`grid.native_lat_T_deg` / `z_coord.nemo_gdept_0` before the perturbation is
built.  (Rule 1e: two independent readings of the same operand must converge
before either is used.)  **Measured:** the card and NEMO's mesh agree to
`0.0` in latitude and `0.0` in depth, with `0` of 704 rounded latitudes and
`0` of 21120 rounded depths differing.  A hazard avoided by that check: the
legacy `nemo_gyre_latitudes` builder (`nemo_recipe.py:656`) constructs an
UN-rotated latitude with `sin_alpha = 0`, and a harness that reached for it
would silently force the year with the wrong wind.  The card path reads the
real two-dimensional `gphit`, and the reconciliation is what proves it.

## 4.  What is scored, and the metric

Per scored day, on NEMO's own `tmask` from `mesh_mask.nc`, fp64.  The arrays
are 32x22x30 -- that is the FULL global array (`kpi = 30*nn_GYRE+2`,
`kpj = 20*nn_GYRE+2`, `usrdef_nam.F90:74-75`); 30x20 of those columns are wet
and the closed-boundary ring is masked out of every number below.

| row | quantity | units |
|---|---|---|
| `T3D` | 3-D wet-cell rms difference of temperature | K |
| `T3D_0_100`, `T3D_100_1000`, `T3D_1000P` | the same, in three depth bands | K |
| `S3D` | 3-D wet-cell rms difference of salinity | g/kg |
| `SST` | level-1 rms difference of temperature | K |
| `SSH` | surface-height rms difference | m |
| `PSI_MAX`, `PSI_MIN` | SIGNED extrema of the barotropic streamfunction | Sv |
| `QNET` | area-integrated net surface heat flux | W |

Plus a per-day DIAGNOSTIC, not a verdict row: the number of wet cells whose
temperature difference exceeds 1 mK, within each ensemble and across the two
models.

`T3D` is the headline, so GYRE's number is directly comparable with DINO's.

**Why the depth bands.**  The floor and the gap otherwise live in different
volumes.  The 1e-10 K seed is applied uniformly to the bottom, and below the
thermocline nothing in year 1 moves it; a year-1 from-rest gap lives in the top
few hundred metres.  A whole-column rms therefore dilutes the FLOOR relative to
the GAP by roughly the square root of the volume ratio -- about 3x -- and
biases the verdict toward DISTINGUISHABLE for a reason that has nothing to do
with fidelity.  Both independent reviews of this document raised it.  The bands
cost nothing: they come from the same saved states.

**Why signed PSI.**  A double gyre has a subtropical and a subpolar cell.
`max|PSI|` reports one of them, is blind to the other, and is blind to a sign
flip.

**Why QNET.**  `qns` is a 40 W/m2/K restoring to the analytic `t_star`
(`usrdef_sbc.F90:118-120`), so it is a function of the model's own SST.  That
makes the area integral a budget-closing discriminator no field rms provides,
and it is where a surface-flux or top-cell mismatch appears first.

**Why S3D is kept, and what it measures.**  GYRE has `sfx = 0` and a
zero-mean `emp` (`usrdef_sbc.F90:139-146`), so salinity has no source; its
initial profile is depth-varying (36.25 at the surface to ~35.5 at depth).
`S3D` is therefore a clean PASSIVE-TRACER TRANSPORT row that isolates advection
and diffusion from thermal forcing.  It is not a redundant copy of `T3D`.

**What is NOT scored, and why (Rule 1: waive in writing).**  Western-boundary-
current separation latitude, thermocline depth, and the subtropical/subpolar
transport partition are all better discriminators of a dynamical mismatch than
an rms, and none of them is scored this round.  The reason is scope, not
merit: each needs a definition that would itself have to be preregistered and
validated, and the depth-banded rows plus `PSI_MAX`/`PSI_MIN` already localise
a difference in depth and in gyre.  They are named here so the omission is a
decision on the record rather than an oversight.

Field rows use **pairwise distances, never a difference of means**: an rms is
positive definite and the difference of two rms values is not a distance.  The
scalar rows keep the mean-difference form, which is valid for them.

## 5.  The floor

For each field row and each day:

```
spread_lego = max over the 6 within-ensemble pairs of rms(A_i - A_j)
spread_nemo = max over the 6 within-ensemble pairs of rms(B_i - B_j)
floor       = sqrt(spread_lego^2 + spread_nemo^2)
```

The within-pair statistic is already a DIFFERENCE of two runs, so no extra
`sqrt(2)` is applied (the `sqrt(2)` factor in
`dino_1226/floor90_ensemble.py:228` converts a single-run std into a difference
floor; here the statistic is a difference to begin with).  `max` over pairs is
the sample RANGE, matching `floor90_ensemble.spread`'s max-pairwise convention,
and at n=4 that runs about 2x a std BY CONSTRUCTION — stated so the floor is
never quoted as if it were a standard deviation.

The scalar rows use the SAME max-pairwise range over the four members, with
**no extra `sqrt(2)`**.  An earlier draft applied one "because a gap in a
scalar is a difference of two single runs"; that factor was already inside the
statistic, and both independent reviews caught the double count.  The `sqrt(2)`
at `dino_1226/floor90_ensemble.py:228` converts a SINGLE-RUN standard deviation
into a difference floor, and this statistic is not a single-run std.

**How many sigma the bar actually is, stated so it is never mistaken for one.**
A max over 6 pairs at n=4 is the sample RANGE, about 2 sigma; two of those
combined in quadrature is about 2.8 sigma; the verdict multiplies by 2.  The
`gap <= 2*floor` bar is therefore roughly **3.4 sigma of the models' own
spread**.  Read DINO's `0.360` under that: its day-360 gap was around 1 sigma
of the model's own spread, which is a stronger statement than the ratio alone
suggests.

## 6.  The verdict rule

```
gap_d  = rms( legoESM control at day d  -  NEMO control at day d )
INDISTINGUISHABLE at day d   iff   |gap_d| <= 2 * floor_d
```

The headline is the **crossing day** — the first scored day at which the gap
falls inside `2*floor`, or "never within the year" — not a binary at one
arbitrary day.  The pooled cross-ensemble distance `d(A_i, B_j)` is reported as
a supporting column.  There is no permutation test: 4 vs 4 gives 35 distinct
splits, so the smallest two-sided p is `0.0286` over ~5 correlated rows, and it
could not carry a verdict.

## 7.  §1 — the bounded-offset test

DINO's §1 asked whether a flat gap against a growing floor is a bounded
deterministic offset rather than a diverging trajectory.  The same
classification is applied here, on the measured record:

```
G_gap   = gap_360   / gap_90
G_floor = floor_360 / floor_90
```

| condition | classification |
|---|---|
| `G_floor / G_gap >= 10` | **BOUNDED_DETERMINISTIC_OFFSET** — the verdict at day 360 is a statement about when a growing floor overtook a flat gap, not about fidelity; the crossing day is the whole deliverable |
| `0.1 < G_floor/G_gap < 10` | **CO_GROWING** — gap and floor are the same kind of object; the ratio is meaningful |
| `G_floor / G_gap <= 0.1` | **GAP_AMPLIFYING** — the gap grows while the floor does not; this is the signature of a real operator mismatch and names GYRE's remaining owner as the next target |

**Where GYRE's protocol necessarily departs from DINO's.**  DINO's §1 was a
PRE-CHECK that could cancel the experiment, because DINO's gap was already
measured and the only unknown was the floor.  **GYRE has no prior gap** — no
NEMO GYRE year exists — so nothing can be cancelled on a ratio before Phase 1.
Phase 0 is still run first, and still first, because Rule 3 says calibrate the
instrument before quoting a residual; but its gate here is a **vacuity** gate,
not an economy gate:

* `floor > 0` at every scored day (a zero floor means the seed never reached
  the state, and would be read as "distinguishable" — the exact false positive
  this harness exists to avoid);
* the four legoESM members are pairwise distinct at day 360.

The DINO economy argument does not bind on GYRE anyway: a NEMO GYRE year is
2160 steps on a 32x22x30 domain, and the certified 10-step runs take 1.9 s
wall including start-up, so four NEMO members cost minutes, not a budget.

## 8.  PREREGISTERED EXPECTATIONS — falsifiable, committed before any member runs

All are **PLAUSIBLE**: one measured entering condition (section 0) plus scaling.
Each names what refutes it.

**P1 — the floor stays far below the gap.**  legoESM's day-360 `T3D` ensemble
floor is **below 1e-3 K**.  *Reason, REWRITTEN after review — the first
version's mechanism was wrong and is retracted:*  the defensible laminarity
claim is the DEFORMATION RADIUS, not the Munk layer.  A first-baroclinic speed
of ~3 m/s gives Rd of 27-80 km across the basin against a 106 km grid, so
baroclinic instability is unresolved everywhere and there is no resolved eddy
field to grow a perturbation exponentially.

The first version argued from the Munk layer: `rn_Uv = 2.0`, `rn_Lv = 100 km`
give `A_hm = 1e5 m2/s`, so `(A_hm/beta)^(1/3) = 171 km` at 29N.  That
arithmetic is right and both reviewers confirmed it, but the conclusion drawn
from it was not: 171 km on a 106 km grid is **1.6 cells, which is UNDER-
resolved**, and an under-resolved western boundary current is a noise SOURCE,
not a guarantee of smoothness.

**And the laminar argument does not bound the real candidate amplifier.**
`ln_zdfevd = .true.`, `rn_evd = 100.`, `nn_evdm = 1` are ON in the certified
namelist.  `zdfevd.F90:93` is a hard branch -- `IF( MIN(rn2,rn2b) <= -1.e-12 )
p_avt = rn_evd` -- a 1e7 jump from `rn_avt0 = 1.2e-5`, applied to momentum as
well.  A threshold process converts a rounding-level perturbation into a finite
difference in ONE cell in ONE step, and GYRE convects hard: the 40 W/m2/K
restoring reaches a `t_star` of about -2.6 C at 50N in winter against an
initial SST of 23.5 C.  So P1 is a prediction about how OFTEN that switch flips
differently between members, which no laminar argument can settle in advance.
The per-day cell-count diagnostic (section 4) is how it is watched.
**REFUTED if `floor_360(T3D) >= 1e-3 K`.**

**P2 — the floor's growth is far below DINO's.**  `floor_360/floor_30 < 1e3`
for `T3D`; DINO's was 438 in a basin that develops eddies.
**REFUTED otherwise.**

**P3 (CALIBRATED, not blind — see the label note below) — the gap does not
come back down.**  The control-vs-control `T3D` gap is
**at least `2.8e-3 K` at every scored day** (its already-measured 40-hour
value) and **at most 3 K** (the field's own scale: two unrelated GYRE spin-ups
could not differ by more).  **REFUTED if any scored day falls below 2.8e-3 K,
or if day 360 exceeds 3 K.**  The lower edge is the real content: a gap that
*shrinks* over the year would mean the year-scale solution is set by the
forcing and not by the trajectory, and would make the ten-step DEBT rows
climate-irrelevant.

**P4 (CALIBRATED) — the verdict.**  GYRE is **DISTINGUISHABLE at every scored day**, and the
crossing day is **"never within the year"**: `gap_d > 2*floor_d` for
`d = 30 ... 360`.  *Reason:* the gap is already 2.8e-3 K at 40 hours while the
ensemble's seed is 1e-10 K, so the floor would have to gain seven orders inside
a year on a flow P1 argues cannot supply two.  **REFUTED if any scored day
gives `gap_d <= 2*floor_d`.**

Note P4 predicts the OPPOSITE of DINO's outcome (`gap/(2*floor) = 0.360`,
INDISTINGUISHABLE).  That is the point: a preregistration that predicts the
previous case's answer is not a prediction.

**P5 — the shape, which is what names an owner.**  The §1 classification comes
out **GAP_AMPLIFYING** or **CO_GROWING**, not `BOUNDED_DETERMINISTIC_OFFSET`.
**REFUTED if `G_floor/G_gap >= 10`.**

**What would make this round UNINFORMATIVE, said in advance.**  If P4 holds by
six orders of magnitude at every day, the year adds little beyond the ten-step
ladder except the MAPS: where the difference lives at day 30 and day 360, and
whether it is concentrated in the western boundary current, the thermocline, or
the surface-forced layer.  Those maps are therefore a deliverable of this
round, not an optional extra, and they are produced whatever the verdict says.

**LABEL NOTE, raised by review and accepted.**  P1, P2 and P5 are blind: the
floor has not been measured.  P3 and P4 are **CALIBRATED**, because the
40-hour gap in section 0 was measured while this document was being written and
the earlier band was replaced after seeing it.  Calling them predictions would
be a postdiction in a file labelled PREREG.  They are still falsifiable and
still committed before any member runs, but they are not evidence of
foresight and are not reported as such.

**AND THE VERDICT IS NOT THE DELIVERABLE IF THE FLOOR COLLAPSES.**  One review
made the structural objection that this floor is an INITIAL-CONDITION
perturbation spread, while the owner's bar is each model's own RUN-TO-RUN
spread.  On a flow that damps an IC perturbation those are different numbers,
and the IC floor can shrink toward zero -- at which point every operator
difference reads DISTINGUISHABLE for a reason that has nothing to do with
fidelity, and P4 cannot fail.  So the harness classifies the floor's own shape
(`FLOOR_COLLAPSE` vs `FLOOR_GROWING`) and, when it collapses, the round reports
the maps and the reproducibility floor and explicitly does NOT report a
fidelity verdict.  The same-binary reproducibility floor (section 9) is added
for exactly this reason.

### A retraction this document already owes

An earlier draft of this section, written before the ten-step pre-measurement,
predicted a day-360 `T3D` gap in `[1e-12 K, 1e-4 K]` and reasoned from the
kt=2 rows as if they were the seed of an otherwise-shared trajectory.  The
measured 40-hour gap is `2.768e-3 K`, four to nine orders above that band.
**That band is RETRACTED before it was ever tested**, and the retraction is
kept here rather than deleted because the error it records — quoting an early
ladder row as if it bounded a long run — is the one this protocol exists to
prevent.

## 9.  Phase order

* **PHASE 0 — legoESM only, no NEMO time.**  Control + 3 members, 360 days,
  scored at days 30/60/.../360.  Writes `phase0_floor.json`.  Cost: 2160 steps
  x 0.478 s/step = **17.2 min per member** on CPU, four members in parallel.
* **PHASE 0b — the same-binary reproducibility floor.**  Seed 0 re-run under a
  different XLA thread count, into `lego_seed0_repro`.  This is the floor the
  owner's phrase "each model's own run-to-run spread" literally names, and it
  is a DIFFERENT number from the IC-perturbation floor whenever the flow is not
  chaotic.  Cost: one more member-year.  If it comes out exactly zero, that is
  itself the finding -- the model is bit-reproducible and the IC floor is the
  only floor available, which must then be said rather than assumed.
* **PHASE 1 — NEMO.**  The four members via `run.sh`, which the operator runs;
  the agent writes it and stops.  `run.sh` refuses unless `phase0_floor.json`
  exists and passes the vacuity gate.
* **PHASE 2 — score.**  Verdict table per day, §1 classification, figures,
  SHA-256.

### CPU is not a choice here — the card forbids the GPU

`validate_nemo_testcase_card` requires `transcendentals == "libm"`
(`ocean/fidelity/nemo_testcase_recipe.py:867-871`), and
`PrecisionPolicy` raises `transcendentals='libm' is CPU-only` on any other
platform.  A GPU run of this card is not slower, it is **impossible**, so the
GPU/CPU question this round could otherwise have to answer does not arise, and
no 10-step GPU-vs-CPU bit comparison is owed.  Measured and recorded, not
assumed: the GPU launch of the card's own step loop exits with that error.

## 10.  What this cannot see

* **The floor measured is an INITIAL-CONDITION perturbation spread, not
  run-to-run irreproducibility.**  Those are the same number only on a chaotic
  flow.  Phase 0b measures the second one; until it reports, every ratio in
  this document is a ratio against the first.
* Four members is a small ensemble.  At n=4 the relative standard error of a
  spread estimate is `1/sqrt(2(n-1)) = 41%`
  (`dino_1226/floor90_ensemble.py:71`), and that sampling error is **not**
  propagated into the verdict.  A ratio between 0.5 and 2 is reported as
  `MARGINAL`, not as a verdict.
* **The whole-column rows compare a floor and a gap that live in different
  volumes** (section 4).  The depth-banded rows are the mitigation, not a cure:
  within a band the same argument applies at smaller amplitude.
* Day 360 is 26 December on this calendar, the seasonal MINIMUM of the forcing
  and the maximum of convection, i.e. the phase at which the `ln_zdfevd` switch
  is most active.  Day 180 is the opposite phase.  A verdict read off day 360
  alone is read at one extreme of a seasonal cycle.
* Day 360 is also only about a third of the first-baroclinic basin-crossing
  time, so the year is an early part of an adjustment, not an equilibrium.
  Nothing here says what decade 3 does.
* Only the rows of section 4 are scored.  A difference living entirely outside
  them is invisible by construction (Rule 1's blind spot), and the four
  discriminators named there as NOT scored are the most likely such place.

## 11.  ASKED / UNASKED

Measurement-neutral means the choice cannot move a scored number.

| # | choice | value | ASKED? |
|---|---|---|---|
| 1 | run length | 2160 steps = 360 days | **ASKED**, pick: 2160.  An earlier draft called this forced; that was RETRACTED (section 2) -- the forcing's `nyear` term is a no-op at every kt, so any length is transcribable.  2160 is one exact seasonal cycle and puts GYRE's scored days on DINO's |
| 2 | snapshot / restart cadence | every 180 steps = 30 days | **ASKED**, pick: 30 days, matching DINO's scored days |
| 3 | number of members | control + 3, per model | **ASKED**, pick: 4, matching `floor90_ensemble.SEEDS = (None,1,2,3)`; the 41% spread error it implies is in section 10 |
| 4 | seeds | 0,1,2,3 | **ASKED**, pick: these; seed 0 doubles as the byte-identity control on the patch |
| 5 | `nn_write` | 2160 | measurement-neutral (output cadence; scoring reads restarts) |
| 6 | `cn_exp` | unchanged | measurement-neutral; keeps the namelist diff at three rows plus one addition |
| 7 | CPU vs GPU | CPU | **not a choice** -- the card requires `transcendentals='libm'` and the precision policy raises `'libm' is CPU-only` on any other platform.  Measured: the GPU launch of the card's own step loop exits with that error |
| 8 | scored rows | T3D + 3 depth bands, S3D, SST, SSH, signed PSI_MAX/PSI_MIN, QNET, plus a 1 mK cell count | **ASKED**, pick: these.  The campaign scores no GYRE metric today (`REGISTERED_METRICS`, `nemo_testcase_full_statistics.py:87-110`, covers only the two tanks).  All are computed from saved states and change no run.  Four better discriminators are named in section 4 as deliberately NOT scored |
| 9 | perturbation amplitude / expression | DINO's, verbatim | **not a choice** -- copied from `cfgs/DINO/MY_SRC/usrdef_istate.F90:177-183` so the two cases share one instrument |
| 10 | zonal-uniformity assertion | REPLACED by `NINT(gphit*1000)` group constancy | **not a choice** -- DINO's assertion is false on GYRE's rotated grid and would be vacuous or red.  Measured on the card: 1470 groups, 1470 distinct values, zero within-group spread |
| 11 | the one non-additive patched line | `NAMELIST/namusr_def/` | **ASKED**, pick: patch it and enforce "exactly one removed line, and it is that one", rather than inventing a second namelist group GYRE does not have |
| 12 | the same-binary reproducibility member (Phase 0b) | run it | **ASKED**, pick: run it.  It is the floor the owner's own words name, it costs one member-year, and without it the verdict rests on a floor that may be measuring perturbation decay |
| 13 | using the existing ten-step NEMO restart to pin the time level and to fix P3's band | used | **ASKED**, pick: use it; it is existing data and costs nothing, but it makes P3/P4 CALIBRATED rather than blind, and they are labelled that way |

UNASKED list: **empty**.

## 12.  Reviews of this document, before any member ran

Two independent fresh reviews were run on the preregistration BEFORE any
simulation -- one on the software and bookkeeping claims, one on the physics
and measurement design.  Codex was occupied on this branch's kt=2 ladder and
GLM-5.2 was unavailable, so both reviewers were Claude instances with no shared
context.  Both found real defects; this document is the revised version.

| finding | reviewers | what changed |
|---|---|---|
| "2160 steps is the only defensible year" is arithmetically false; the `nyear` term is a no-op because the forcing period equals the year | BOTH, independently | section 2 rewritten with an explicit retraction; run length moved to ASKED |
| the `sqrt(2)` applied to the scalar floor double-counts a factor already inside the pairwise statistic | one, with the other reaching the same conclusion via "the bar is ~5.7 sigma" | deleted from the harness; section 5 states the bar is ~3.4 sigma |
| P1's laminarity mechanism is the wrong one, and `ln_zdfevd`'s convective switch is a threshold amplifier no laminar argument bounds | one; the other independently noted 171/106 = 1.6 cells is UNDER-resolved | P1's reason rewritten to the deformation radius and the EVD switch named; a cell-count diagnostic added |
| the floor and the gap are measured over different volumes, biasing toward DISTINGUISHABLE | one | three depth bands added to the scored rows |
| `max\|PSI\|` is blind to the second gyre and to a sign flip; no integral heat-flux discriminator | one | signed `PSI_MAX`/`PSI_MIN` and `QNET` added |
| the floor is an IC-perturbation spread, not run-to-run spread; on a damped flow it collapses and P4 cannot fail | one | Phase 0b reproducibility floor added; `FLOOR_COLLAPSE` classified; the verdict is explicitly withheld in that case |
| P3/P4 were fixed after seeing the 40-hour measurement -- a postdiction in a PREREG | one | relabelled CALIBRATED, with the reason stated |
| `pdept` is `gdept(:,:,:,Kbb)`, equal to `gdept_0` only because `ssh = 0` at init | one | section 3 states the condition |
| `nemo_gyre_latitudes` builds an UN-rotated latitude and is a landmine for any harness that reaches for it | one | section 3 records it; the operand reconciliation is what would catch it |
| "interior 32x22x30" mislabels the full global array | one | section 4 corrected: 32x22 is the array, 30x20 the wet columns |
| S3D was called potentially vacuous; it is a clean passive-tracer transport row | one (correcting the other's premise) | section 4 says what it measures |
| day 360 sits at the seasonal convection maximum and at ~1/3 of the basin-crossing time | one | section 2 and section 10 |
| `ndate0` was inferred, not read | one | section 2 reads it off the run's own output filename |

Both reviewers said, in different words, that the year as designed answers "is
GYRE chaotic?" more sharply than "do the two models agree".  That objection is
not fully answered by any change above -- it is answered by Phase 0b, by the
`FLOOR_COLLAPSE` classification, and by treating the MAPS as a deliverable
regardless of the verdict.

## 13.  Retraction already owed by this document

An earlier timing of legoESM's GYRE step in this session reported
**7.15 s/step**, implying 4.3 h per member-year and motivating a GPU arm.  That
number averaged over the compile-and-retrace tail of the first ten steps.  The
steady-state cost, measured after the cache is warm, is **0.478 s/step**
(17.2 min per member-year).  The 7.15 s/step figure is **RETRACTED**; no GPU
arm is needed, and section 9 records that the card could not have taken one
anyway.
