# The free-slip southern wall in the lateral-momentum viscosity: NEMO against legoESM

Follow-on to `dino_basin_budget_result.md` Part 4, which established that the
southern-basin deficit lives in four rows against DINO's southern wall and that
the flow there is 91% geostrophic, and which then registered two next actions: a
line-by-line comparison of how each model's lateral-viscosity operator meets the
free-slip wall, and a one-variable viscosity ablation. This is both, plus a third
arm that the first two turned out to need.

Written for a reader who has not followed the campaign.

Probes, both committed, both carrying two-directional self-tests:
`scripts/validate/ocean_fidelity/dino_1226/wall_ldf_alignment.py` and
`wall_visc_ablation_gap.py`. Pre-registration and its addendum, each written
before the arms it governs were launched: `PREREG_wall_viscosity_ablation.md`.
Both probes and both readings passed two independent adversarial reviews, and
the review changed the answer — see "What the reviews overturned".

---

## The headline

1. **The wall treatment is identical in the two models, measured rather than
   argued.** legoESM's corner mask — the thing that decides whether the wall
   corner can carry a viscous stress — equals NEMO's `fmask` at every one of the
   372528 corner points, on all 36 levels, zero disagreements. The wall corner
   carries exactly zero stress on both sides. Form, coefficient, corner mask,
   wall flux, and every metric the operator divides by all agree, the last of
   them to 4.2e-5. **There is nothing in this operator's wall treatment to fix.**
2. **The two differences that do exist inside the operator are both far too
   small.** NEMO's layer-thickness weighting changes the viscous tendency by
   1e-4 relative — and on DINO's full-step grid the thickness is horizontally
   uniform within every level, so the weighting is provably near-inert, not just
   small. The 2-D-versus-3-D masking changes it by exactly zero, because the
   model's velocity below the sea floor is exactly zero. The defect they would
   have to explain is 34%.
3. **Friction is a strong lever on the wall rows and is NOT their owner.** This
   is the pre-registered discriminator and it came back clean. Raising the
   lateral viscosity by 1.5× and 2× makes the southern basin's error grow
   *monotonically and almost linearly* — the unsigned row error goes 0.446,
   1.141, 1.800 Sv, and the 1.5× point sits 1.6% off the straight line joining
   the ends. That is the registered signature of a viscosity that moves someone
   else's error around, not of a viscosity that is set wrong.
4. **The earlier reading of the two-arm pair is RETRACTED.** The first draft of
   this document said doubling the viscosity shrinks the wall rows' error by 85%
   and confirmed friction as the owner. The 85% is a signed sum over four rows
   whose error becomes dipolar inside that window: every individual row got
   worse and they cancel. Unsigned, the wall rows are 2.2× worse and the whole
   southern band 4.0× worse. **The CONFIRM is withdrawn.**
5. **The Munk length-scale argument that promoted friction in the first place
   does not survive.** `(A/β)^(1/3)` is derived for a *meridional* wall; DINO's
   southern wall is zonal, and in a zonally closed basin the β term never enters
   the balance. The 87 km it produced is arithmetic. And the measured 2.69-row
   width is not evidence either: the deformation radius here is about 9 km
   against a 40 km cell, so any wall-trapped structure is resolution-limited to
   two or three cells whatever made it.
6. **The one thing that got better under more viscosity is a compensating
   error.** The circumpolar transport error falls monotonically (0.382 → 0.242 →
   0.069 Sv) while the density contrast that sets it degrades monotonically
   (2.4e-4 → 7.6e-4 → 1.4e-3 kg/m³, failing its gate from 1.5× on). You cannot
   make a thermal-wind transport more accurate while making the density field
   behind it less accurate unless two errors are cancelling.
7. **Nothing here should be shipped.** NEMO runs `rn_Uv = 0.27` and so does the
   card; the raised-viscosity arms are diagnostic only and are less faithful by
   construction.

**One correction and one clarification to the numbers this lane inherited.** The
four wall rows carry **67.7%** of the day-90 basin gap (−0.288 of −0.426 Sv), not
95%; 95% is the **day-360** figure and the two were being conflated in the
hand-over. Separately, "the channel sits at 0.15 noise floors" is a **day-360**
number on one particular reduction of the channel-band transport against its own
0.0101 Sv floor (`dino_verdict360_result.md:49`); the 90-day acceptance gate
scores a different reduction against a 0.091 Sv floor and reads 0.382 Sv on the
shipped card. Different measurements of the channel, not a contradiction.

---

## Part 1 — the alignment table

DINO's southern boundary is a land row (grid row 0, −69.85°, dry at every level;
verified, not assumed). The first wet row is row 1 at −69.50°, and rows 1–4 are
the four "wall rows". They sit on a bathymetric staircase: 30–31 wet levels at
row 1 rising to 34 by row 4, against 35 in the interior.

Oracle: NEMO 5.0.2 at `~/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/`. legoESM
paths are relative to the repository root.

| # | what | NEMO 5.0.2 (DINO) | legoESM (`nemo_dino_kamm_mlf`) | verdict |
|---|---|---|---|---|
| 1 | operator the configuration selects | `ln_dynldf_lap=.true.`, `ln_dynldf_lev=.true.` (`RUN_VERDICT360_M0/namelist_cfg:365-366`) → `dynldf_lev.F90:45` `dynldf_lev_lap` | `"lateral_viscosity_operator": "nemo_div_curl"` (`dino.py:1324`) → `latlon_cgrid_operators.py:1299` | **MATCH** — harmonic, iso-level |
| 2 | operator form | vorticity–divergence, `grad(ahmt·div) − curl(ahmf·curl)`. `nn_dynldf_typ=0` is the default (`namelist_ref:1108`) and DINO does not override it; branch at `dynldf_lev.F90:86` | same form and same signs, `latlon_cgrid_operators.py:1390-1391` | **MATCH** |
| 3 | the coefficient | `ahmt = ½·rn_Uv·max(e1t,e2t)`, `ahmf = ½·rn_Uv·max(e1f,e2f)` (`ldfdyn.F90:251`, `:286`; `ldfc1d_c2d.F90:138-139`), `rn_Uv=0.27` (`namelist_cfg:369`) | same formula, `latlon_cgrid_operators.py:1237`; `½·rn_Uv` recovered as `A_h/(R·Δλ)` at `ocean_pe_latlon_cgrid.py:2750` from `dino.py:3020` with `U_M=0.27` (`dino.py:665`) | **MATCH** — T-points bit-identical at the surface, F-points to 1.5e-5 (the discrete `max(e1,e2)` convention, already documented) |
| 4 | what the wall corner sees | the circulation `δi(e2v·v) − δj(e1u·u)` at the f-point (`dynldf_lev_rot_scheme.h90:23-25`) | the same circulation, `curl_vertex_cgrid` (`latlon_cgrid_operators.py:1380`) | **MATCH** — both impose free slip as **zero relative vorticity at the coastal corner** |
| 5 | how free slip is imposed | `rn_shlat=0` (`namelist_cfg:198`). `fmask` = product of the four surrounding `tmask` (`dommsk.F90:152`), zero at every corner touching land; the block that would raise it (`dommsk.F90:209-217`) does not run. Folded into the coefficient once at init: `ahmf *= fmask` (`ldfdyn.F90:330`) | the same four-cell product (`operators_latlon_cgrid.py:1570`), applied to the vorticity inside the operator (`latlon_cgrid_operators.py:1384`) instead of folded into the coefficient | **MATCH — 0 disagreements out of 372528 corner points, on every one of the 36 levels.** An off-by-one in either index would have shown 3718 (row shift) or 11584 (column shift), so the check discriminates |
| 6 | staircase corners below the sea floor | `fmask` is genuinely 3-D (`ldfdyn.F90:330` multiplies level by level) | a 3-D corner mask built per level from the coordinate's activity flag (`ocean_pe_latlon_cgrid.py:2753-2769`) | **MATCH** — included in the count above |
| 7 | viscous stress crossing the wall face | zero: the wall corner's coefficient is zero, so the curl term at the first wet row (`dynldf_lev_rot_scheme.h90:41`) draws nothing from the wall side | zero by the same construction | **MATCH — measured exactly 0.0 on both sides**, as the coefficient *and* as the stress the operator forms on the real state |
| 8 | the divergence term at the wall | `ahmt` carries `tmask` (`ldfdyn.F90:329`) | the divergence is masked before the coefficient (`latlon_cgrid_operators.py:1374`) | **MATCH** in form |
| 9 | mask dimensionality | 3-D on every leg | the production call hands the operator the **2-D** cell and face masks (`ocean_pe_latlon_cgrid.py:4246-4247`) where every other momentum term in the same routine gets the 3-D ones | **DIFF in the wiring, ZERO in the answer — see the control below** |
| 10 | layer-thickness (e3) weighting | present: `e3f` on the corner vorticity (`dynldf_lev_rot_scheme.h90:23`), `e3t`/`e3u`/`e3v` inside the divergence (`:27-29`), divided out by `e3u`/`e3v` (`:41,:51`) | absent on the card: `lateral_viscosity_e3_weighting="off"` (`dino.py:834`) | **DIFF in the code, provably near-inert on this grid — see below** |
| 11 | metrics the operator divides by | `e1e2f` on the vorticity (`h90:23`), `e1e2t` on the divergence (`:27`), `e2u`/`e1u` and `e1v`/`e2v` on the two tendencies (`:41-42,:51-52`) | the geometry's own dual-cell area, cell area and four face lengths | **MATCH** — T-cell area and all four face lengths bit-identical; F-cell area and v-face width to 4.2e-5, at the wall rows and everywhere but the two polar end rows, where legoESM deliberately zeroes the dual-cell metric as a wall boundary condition |
| 12 | the outer face mask | `umask`/`vmask` multiply the result (`h90:40,50`) | the same (`latlon_cgrid_operators.py:1393-1395`) | **MATCH** |
| 13 | any extra wall or slope term | none | a slope-foot viscosity enhancement exists but its coefficient defaults to 0 (`state.py:1779`) and the card never sets it, so it is a multiply by 1.0 | **MATCH** — nothing extra runs |

### Row 9, and the control that makes its zero mean something

Applying the 2-D and the 3-D maskings to the same state returns exactly zero on
every wet face. That number on its own says nothing, because the state's
velocity is already zero on the 15017 faces the two maskings disagree about, so
those faces contribute nothing either way. The probe therefore plants 0.5 m/s on
exactly those faces and re-runs both arms: the difference becomes 4.08e-6 m/s²,
2.6 times the wet-face tendency, and the probe aborts if it cannot separate them.

So the honest statement is not "the two maskings agree" but: **legoESM's velocity
is exactly zero below the sea floor — on the bridged restart and on the day-90
state of every arm — so the operator's 2-D masking is indistinguishable from
NEMO's 3-D masking in this configuration.** The companion reader re-checks that
on every arm it scores, because it is a property of the state and states change.

One piece of real debt falls out: the production arm *does* write a viscous
tendency on those below-seafloor faces, up to 1.5e-7 m/s². Nothing downstream
reads it, so it is latent, not live.

### Row 10, and why the thickness weighting cannot be the owner

Two independent arguments, both measured.

*Empirically*: applying both treatments to the identical state changes the
viscous tendency by **1e-4 relative** at the wall rows (2.4× enriched over the
interior, but 1e-4 either way), and the largest relative change anywhere in the
basin is also 1e-4. The defect is 34%. Three orders of magnitude short.

*Structurally*, which is the stronger argument: DINO is a **full-step**
z-coordinate (`namelist_cfg:70-71`, `ln_zco_nam=.true.`, `ln_zps_nam=.false.`).
Reading the mesh directly, `e3t_0`, `e3u_0`, `e3v_0` and `e3f_0` are **identical
to each other and horizontally uniform within every level**, spread exactly 0.0
over the wet cells at all 36 levels. Every e3 factor in NEMO's formula therefore
appears as a ratio of equal numbers and cancels identically; what is left is set
by the sea surface through `η/H`, which at these rows is 4e-4 and varies across
a stencil by about 1.6e-6. **On this grid NEMO's thickness weighting is very
nearly the identity.** A recorded A/B found legoESM's faithful thickness-weighted
variant *worsens* the single-step agreement — if NEMO's version is near-inert
here, that A/B is a bug report against the variant's own transcription, not
evidence about NEMO. Leaving the weighting off is right, for a better reason
than the A/B gave.

### What this table does not cover

There are **two** wall conditions in this model, and this table is one of them.
The vorticity term has its own: NEMO sets `ln_dynvor_msk = .false.`
(`namelist_cfg:333`), keeping the coastal relative vorticity alive in the
vorticity flux, and the card matches that by configuration elsewhere
(`dino.py:1491`, `een_q_boundary="nemo_live"`). It is aligned, but it is aligned
somewhere else, and a reader who concludes "the wall is matched" from this table
alone would be over-reading it.

---

## Part 2 — the ablation, and what it says

One variable: the lateral viscous velocity `rn_Uv`, at 0.27 (the oracle's own
value), 0.405 and 0.54. Everything else byte-identical — same commit, same NEMO
day-180 restart, same 90 days, same vertical ladder, same everything. The runner
stamps the viscosity each arm actually ran at into its own output file, and the
reader labels arms from that stamp rather than from a filename. The runner's
smoke check diffs all 206 leaves of the built model configuration and requires
the knob to move exactly one of them, so "one variable" is a checked claim and
not an assertion.

**The instrument reproduces the recorded numbers before it is used.** The 1×
arm returns a day-90 basin gap of −0.4258 Sv against the parent document's
−0.426, and a wall-anomaly e-folding scale of 2.69 rows. That is an independent
reproduction, not a re-read: both sides are different runs from the parent's — a
fresh twin against NEMO's own 90-day continuation, where the parent used the
four-member year-long pair.

### The registered discriminator

The addendum registered, before the 1.5× arm ran: the **unsigned** row error
across the southern band separates "legoESM's realized dissipation is genuinely
too weak" (in which case it dips below the 1× value on the way to 2×) from "the
viscosity is a lever on someone else's error" (in which case it climbs
monotonically and the signed sum crosses zero purely by cancellation).

| day 90, legoESM − NEMO | rn_Uv 0.27 | 0.405 | 0.54 |
|---|---:|---:|---:|
| four wall rows, signed sum [Sv] | −0.2884 | −0.1150 | +0.0444 |
| four wall rows, **sum of magnitudes** [Sv] | 0.2884 | 0.3793 | 0.6358 |
| southern band rows 0–13, signed [Sv] | −0.4258 | −0.4341 | −0.4180 |
| **southern band, sum of magnitudes [Sv]** | **0.4459** | **1.1407** | **1.8002** |
| sea-surface excess at the wall row [mm] | 1.722 | 1.829 | 1.613 |
| circumpolar transport error [Sv] | 0.3823 | 0.2416 | 0.0686 |
| upper density contrast error [kg/m³] | 2.40e-4 | 7.65e-4 | 1.43e-3 |
| 5-metric acceptance gate | **5/5 PASS** | 4/5 | 4/5 |

The discriminator: 1.1407 against a straight line between the endpoints at
1.1231 — **1.6% above it, inside the registered 10% band, and monotone.**

**Registered verdict: the lateral viscosity is a LEVER on the wall error, not
its OWNER.** It rearranges the error meridionally — strongly, far beyond any
noise floor — without reducing it. The band's signed total is nearly invariant
across a factor of two in viscosity (−0.4258, −0.4341, −0.4180), which is close
to a property of the operator: with free slip at the wall, the
meridionally-integrated lateral friction telescopes to a single stress at the
band's northern edge, so there is almost no interior route by which the
coefficient can change the total. Since the defect being chased *is* a
band-integrated deficit, that is an argument against friction owning it.

### The compensation reading, on the same three arms

The transport improves monotonically while the density contrast that sets it
degrades monotonically, failing its gate from 1.5× on. The registered
compensation test asked for an interior optimum in the transport; there is none
in the range tested, so that clause returns nothing. What is measured is a clean
anti-correlation across three points, which is the compensation signature and is
reported as **PLAUSIBLE** rather than confirmed. The mechanism is not mysterious
— more viscosity damps the channel's marginally-resolved eddies, which changes
the eddy heat flux and hence the isopycnal slope — but the mechanism was not
tested here.

A separate finding worth carrying out of this lane: **the campaign's
circumpolar-transport metric is tunable by a knob that degrades the physics
behind it.** That is true independently of anything about the southern wall.

### What the reviews overturned

Two independent adversarial reviews ran on the probes and the readings. Between
them they overturned four things, all corrected above and all recorded in the
pre-registration addendum rather than quietly reinterpreted:

* the signed four-row sum, which turned a 120% degradation into an "85%
  improvement" and produced a CONFIRM that is now withdrawn;
* the e-folding scale of the raised-viscosity arms, which was an interpolation
  run over a profile that changes sign — it now returns "no e-folding" for such
  a profile, with a self-check that proves it still measures a real decay;
* the Munk prediction, which was the wrong scaling law for a zonal wall rather
  than a mis-estimate;
* the mask A/B's zero, which was a property of the state and is now stated as
  one, with a control that proves the probe can see the difference.

They also closed the table's one real gap (row 11, the operator's internal
metrics, which had not been compared at all) and added the missing artifact
stamp, the missing land-mask gate, and the missing "and nothing else" half of
the one-variable check.

---

## Part 3 — the leap-frog composition: the last factor-of-two candidate, refuted

With the operator matched to 1e-4 and its coefficient shown to be a lever
rather than the owner, one candidate remained that is *naturally* a factor of
two: the leap-frog itself. If legoESM applied the friction increment over Δt
where NEMO applies it over 2Δt, or read the wrong time level, the realized
damping would differ by exactly the kind of factor the ablation was groping at.

Probe: `friction_timestep_check.py`.

### Both chains, end to end

| | NEMO 5.0.2 (DINO) | legoESM (`nemo_dino_kamm_mlf`) |
|---|---|---|
| which velocity the operator reads | the **before** level: `dyn_ldf(kstp, Nbb, Nnn, uu, vv, Nrhs)` (`stpmlf.F90:319`), and the scheme takes `pu_in(...,Kbb)` (`dynldf_lev_rot_scheme.h90:24-25,28-29`) | the **before** level: `_ldf_state=(T_before, S_before, u_before, v_before)` (`ocean_model_latlon_cgrid.py:8516-8518`), routed onto the friction call alone (`ocean_pe_latlon_cgrid.py:4241-4242`) |
| where it accumulates | `Krhs` | the withheld dissipative increment (`ab2_scope="advective"`) |
| the timestep | `rDt = 2·rn_Dt = 5400 s` from the second step on (`stpmlf.F90:686`; `:135-136` is the single Euler start), `rn_Dt=2700` (`namelist_cfg:116`) | `rdt = 2.0·dt = 5400 s` (`ocean_model_latlon_cgrid.py:8478`), and the increment is `dt_mom · du_diss` with `dt_mom = dt/dt_mom_ratio` (`:3478`, `:3424`), `dt_mom_ratio` **measured = 1.0** on this card |
| what it is added to | the **before** velocity: `puu(Kaa) = (puu(Kbb) + rDt·puu(Krhs))·umask` (`dynzdf.F90:137-142`), with **no** thickness weighting — DINO sets `ln_dynadv_vec = .true.` (`namelist_cfg:321`) | the **before** velocity: `u_naa = (ubc_bef + (ubc_exp − ubc_now)) + du_diss_bc + btu_exp` (`:8548`), the depth mean travelling separately (`:3905-3911`) |
| the time filter | the **plain** `puu(Kmm) + rn_atfp·(puu(Kbb) − 2·puu(Kmm) + puu(Kaa))` (`dynatf_qco.F90:165-166`, the `ln_dynadv_vec` arm selected at `:162`), `rn_atfp = 0.1` (`namelist_ref:73`) | the identical expression (`ocean_model_latlon_cgrid.py:8322-8324`), `asselin_gamma = 0.1` (`dino.py:1490`) |

### Measured, not read

*Reviewer note, incorporated.* The first version of this measurement divided the
dissipative increment by the operator's own tendency and recovered 5400 s. That
quotient is `dt_mom` **by construction** — the numerator is `dt_mom·f(A_h)` and
the denominator is `f(A_h)` — so it confirms one multiplication and the value of
`dt_mom_ratio`, and says nothing about where the increment then lands. Its
"control", which halved the increment by hand, was a NumPy identity that passes
for any input whatsoever. Both are corrected below; the tautological quotient is
kept, demoted and labelled, because it is still the cleanest read of
`dt_mom_ratio`.

**The multiplication.** Median 5400.0000 s over 167702 wet cells. 1182 of them
(0.7%, spread over 197 rows, only 14 in the four wall rows) sit outside 1e-9 of
that — those are cells where the probe's re-derivation of the operator and the
model's own output disagree slightly, not cells running a different timestep.
Separately, `dt_mom_ratio` **cannot** be anything but 1 on this card: the model
refuses it unless the barotropic solver is the rigid lid
(`ocean_model_latlon_cgrid.py:2605`), and this card runs the split-explicit
solver. That half of the question is closed by construction.

**The composition**, which is the question actually posed: does one operator's
worth of friction reach the *after-level velocity* once, multiplied by 2Δt? Only
a whole step answers that, so the probe runs the real leap-frog in both arms and
regresses the difference in the after-level velocity on `2Δt·ldf`.

| regression of the measured step difference on `2Δt·ldf` | slope | residual |
|---|---:|---:|
| depth deviation | 0.9914 | 11.5% |
| **depth mean** | **0.9803** | 8.2% |
| total | 0.9941 | 10.3% |
| **depth mean, four wall rows only** | **0.9987** | — |

Slope 1.0 means the increment lands once, at 2Δt. 0.5 or 2.0 would be the factor
of two. The residuals are the rest of the step responding nonlinearly to the
changed viscosity, which is expected and is not a discrepancy in the slope.

**The depth mean matters more than the deviation here, and it was nearly
missed.** legoESM adds only the *baroclinic* part of the friction increment at
the momentum update and routes the depth mean through the barotropic solver's
slow forcing (`ocean_model_latlon_cgrid.py:8541-8548`), where NEMO applies the
whole increment in `dyn_zdf` and lets the split-explicit solver divide it. Review
flagged that as the one place a fraction of the depth-mean friction could go
missing — which would be a depth-uniform, wall-concentrated loss, i.e. exactly
the shape of the defect. **Measured, it does not go missing:** the depth-mean
slope is 0.98 basin-wide and **0.9987 at the four wall rows**. That hypothesis is
refuted, and it is the most specific mechanism this lane has been able to kill.

**Two controls, both perturbing the model.** Driving the same step at half the
timestep halves the measured multiplier to exactly 2700.0000 s — a hard-coded
timestep anywhere in the chain fails that. And a null arm built at the *same*
viscosity gives a step difference of exactly 0.0 m/s, so the slopes above are
attributable to the viscosity alone.

**The leap-frog composition MATCHES. The last factor-of-two candidate is
refuted.** Together with the operator, the corner mask, the wall flux, the
thickness weighting and the mask dimensionality, lateral friction is now
exonerated end to end: same operator, same coefficient, same wall condition,
same time level, same timestep, same filter coefficient, and the depth mean
arrives.

**A retraction, in place.** An earlier version of this section said NEMO's
momentum time filter is thickness-weighted where legoESM's is plain, and put
the residual at O(η/H) ≈ 4e-4. **Both halves were wrong, and wrong the same
way: I quoted a branch that does not execute.** DINO sets
`ln_dynadv_vec = .true.` (`namelist_cfg:321`), which selects the *plain*
velocity filter at `dynatf_qco.F90:165-166` and the *plain* velocity update at
`dynzdf.F90:137-142`; the thickness-weighted forms I cited (`:190-206` and
`:145-150`) are the `ELSE` arms and are dead code on this card. The two filters
and the two updates are therefore the **identical algebra**, agreeing exactly
rather than to 4e-4. The correction strengthens the match; it was caught by
review, not by me, and it is the same class of error the campaign's own rules
warn about — citing a line without proving its branch runs.

**One condition the multiplier does not cover, met by configuration.** NEMO
overwrites the entire after-level depth mean with the barotropic solver's
answer *after* `dyn_zdf` and *before* the filter (`mlf_baro_corr`,
`stpmlf.F90:709`, called at `:578`, body `:754-765`), so the filter never sees
the raw leap-frog depth mean. Two models agreeing on multiplier, time level and
filter coefficient but disagreeing on whether that overwrite runs would deliver
different effective damping of the depth-mean friction. This card runs it, at
NEMO's position in the step (`dino.py:1714-1715`,
`ocean_model_latlon_cgrid.py:8303`).

---

## The side finding, for the gate-split proposal

Independent of the southern wall, and worth carrying out of this lane on its own:

**The campaign's circumpolar-transport metric is tunable by a knob that degrades
the physics behind it.** Across the three arms, the transport error falls
monotonically (0.382 → 0.242 → 0.069 Sv, a 5.6× "improvement") while the upper
density contrast that sets that transport through thermal wind degrades
monotonically (2.40e-4 → 7.65e-4 → 1.43e-3 kg/m³) and fails its own gate from
1.5× on. A tuner optimising the headline transport number would walk straight up
that gradient and make the model worse.

The acceptance gate already scores both, and on these arms it does its job — it
went from 5/5 to 4/5. The point for the gate-split proposal is narrower: **the
transport metric should not be readable as a standalone score.** Either it is
reported paired with the density contrast that controls it, or a compensation
check is added that fails when the two move in opposite directions. As it stands,
"the channel improved 5.6×" is a true sentence about a worse model.

---

---

## Part 4 — the two ranked discriminators: drag refuted, the vorticity flux CLEARS both legs

Pre-registration: `PREREG_wall_drag_and_een.md` and its addendum. Probe:
`wall_term_discriminators.py`. Both read on NEMO's own one-step single-rank
dump (kt 5761) from the restart the twins start from, with legoESM bridged from
that same restart — bit-identical state on both sides.

**Read the addendum before the numbers.** Dual review overturned the reduction,
the interior sampling, the time level, and four claims in the ranking that
promoted these candidates. Everything below is the corrected version.

### The reduction — the thing that decided both verdicts

The registered bar (5.9e-11 m/s²) was derived for a **persistent depth-mean
acceleration**. My probe scored `mean |difference|` over every wet cell and
level, which is an **upper bound** on that and a loose one. Both terms are now
scored as the **signed, thickness-weighted, zonally-averaged** difference — the
quantity the bar is actually about — with the old statistic and the coherence
ratio printed alongside.

### Bottom drag — REFUTED, overwhelmingly

| | measured |
|---|---|
| coefficient at u-faces | **bit-exact** — 0 of 9758 cells above 1e-15 |
| bottom-level index vs NEMO's | **0 of 9758 disagree**, 0 in the wall rows |
| assembled increment, coherent reduction | wall **3.9e-18 m/s²**, enrichment **0.07×**, **0.00× the bar** |

Seven orders below the bar and anti-enriched. (The earlier published figure of
4.4e-15 was ~1000× too large: the probe had silently fallen back to the *now*
velocity where NEMO uses the *before* level. Corrected and now impossible — the
absence of a before level raises.)

**And drag is refuted by a bound, which is stronger than the measurement.**
Summing *all* active drag sites at the wall gives ~1.3e-10 m/s², so the defect
would need **45% of the entire bottom-drag term**, against a bit-exact
coefficient, an exact bottom index, and a column depth agreeing to 0.5%. That
holds regardless of which single site was measured — which matters, because I
measured only one of four, and not the one that damps the depth mean.

### The EEN vorticity flux — clears BOTH registered legs

| | measured |
|---|---|
| wall rows, coherent reduction | **3.10e-10 m/s²** = **5.25× the bar** |
| far interior | 3.55e-11 |
| **enrichment** | **8.74×** against a 3× bar |
| pointwise relative disagreement | **2.1e-3** at the wall, 4.9e-4 interior |

**Registered verdict: the EEN vorticity flux passes the shape test on both
legs.** Under the correct reduction it is wall-enriched nearly 9× and clears
the magnitude bar 5×. It is also, at 2.1e-3 pointwise, the **worst-matched
operator this campaign has reported** — two orders below the matched-operator
floor set by the lateral-viscosity coefficients (1.5e-5) and the F-cell area
(4.2e-5).

Row structure: the signed difference is +5.5e-10, +4.5e-10, +2.2e-10 on rows
1-3 and flips to −2.5e-11 on row 4 — wall-trapped and single-signed over the
first three rows, then reversing. The wall mean is carried by rows 1-2.

**It is the fix candidate. No fix is built yet**, because the shape test names
the term, not the mechanism, and review named a specific one to test first (see
below).

### Retractions — four of mine, all confirmed false

1. **"Rows 1-4 are a zonally periodic band."** They have land at columns 0 and
   51; the re-entrant channel is rows **14-48**, and rows 1-13 are a **closed
   sub-basin**. The zonal-integral argument that made drag my top candidate does
   not hold on a blocked row. **The argument was void.**
2. **"1.77× differential spin-down gives 0.19-0.34 e-folds."** Those are the
   *absolute* e-folds; the **differential is 0.07** — about 7% against a 34%
   target, roughly 5× smaller than I claimed.
3. **"The vorticity flux agrees to 1.4e-5 relative."** That was max-versus-max
   on one outlier cell. Pointwise it is 2.1e-3 — and the correct number makes
   the term *more* suspect, not less.
4. **My drag site list was wrong twice**: it mislabelled the implicit path's
   barotropic re-add as an explicit bottom-stress source (the genuinely explicit
   site is dead code here), and it omitted the barotropic substep drag — the
   only site that damps the depth mean.

Plus two probe defects, both caught by review: the *now*-for-*before* time-level
substitution above, and a 3.51× enrichment figure that existed only because the
**equator row** — where `f = 0` and the vorticity flux structurally vanishes —
sat in the denominator. Both corrected; the 3.51× is withdrawn.

### The wall's two boundary conditions disagree with each other, faithfully

DINO leaves the coastal f-point **unmasked** in the vorticity flux
(`ln_dynvor_msk = .false.`, `namelist_cfg:333`), so the wall vorticity is the
shear between the first wet row and the land row's stored zero — a
**no-slip-like** condition — while the *same* model's lateral viscosity treats
that corner as **free-slip**. NEMO's own source flags this as unresolved
(`dynvor.F90:890`, "this should be removed when choosing a unique strategy for
fmask at the coast"). legoESM matches both conditions.

Both therefore rest on that stored value being exactly zero, which the probe now
measures on every run for **both** velocity components (0.000e+00). This is why
the wall row is the only place the vorticity flux carries a term interior points
do not — it amplifies an error in the first wet row's velocity straight back
through ζ. It explains the enrichment; it is not itself the origin.

## Where this leaves the campaign

**Lateral friction is exonerated end to end.** Operator form, coefficient,
free-slip corner mask, wall flux, internal metrics, thickness weighting, mask
dimensionality, the time level it reads, the timestep it is multiplied by, and
the time-filter coefficient — all compared, all matching, the largest residual
anywhere being 1e-4 relative. No fix was built because there is nothing to fix,
and the two pre-registered follow-ons that would have built one are cancelled
for want of a difference to implement.

**Also closed.** The length-scale argument that promoted friction: the Munk
formula does not apply to a zonal wall, and the measured 2.69-row width is the
grid's resolution limit for a smooth decaying structure, so it carries no
mechanism information either way.

**The lead is now named, and the list is reframed.** Review supplied a reframe
worth stating first: the error is 91% geostrophic and depth-uniform, so it **is**
a sea-surface-height error — 4.6e-4 m/s at this latitude over a 39 km row is
0.25 mm per row, about **1.0 mm across four rows**, against the **1.72 mm**
excess the ablation table already measured. The transport error and the sea-
surface excess are one object, not two. And rows 1-13 are a **closed sub-basin**,
so the right control volume is that sub-basin — which nobody has drawn.

1. **The EEN vorticity flux — the lead, and the fix candidate.** The only term
   that clears both registered legs (5.25× the magnitude bar, 8.74× enriched),
   and the worst-matched operator the campaign has reported (2.1e-3 pointwise at
   the wall). Its own size means a **sub-percent** transcription error suffices —
   0.28-3.3% of its coherent wall-row value — and the measured disagreement is
   0.21%, the same order.
   *Cheapest next step, offline*: the specific mechanism review named. Under
   this build (`key_qco key_vco_3d`) NEMO's runtime `nn_e3f_typ` branch is dead
   code; what runs is `e3f_vor = E3fv_0·(1+r3f)`, where the reference part is a
   masked average of the **static** `e3t_0` and `r3f` is an **area-weighted,
   entirely unmasked** four-point sea-surface average (`domqco.F90:177-181`).
   legoESM's `nemo_avg` averages the **live** thickness over the **wet count**.
   Those disagree exactly where a vertex has a dry neighbour and where η/H is
   largest — the shallow wall rows. Recompute part 2 with legoESM's `e3f`
   rebuilt NEMO's way; that is the discriminator, and it needs no run.
2. **The wall-row wind stress — the candidate nobody named.** τ is **0.23 mPa at
   row 1** and 2.27 mPa at row 4, against 174 mPa at row 40 — three orders below
   the basin maximum. The acceleration it supplies is only 2-11× the bar, so a
   **9-58% relative error there carries the defect**, and any score normalised by
   the field's RMS (which is how the campaign's "surface forcing matches at 1.0"
   was earned) is structurally blind to it. The analytic profile is a smoothstep
   between nodes at −70° and −45°; at row 1 a 0.25° error in either the latitude
   or the node position is a 100% error at that row.
   *Cheapest discriminator*: diff legoESM's wind stress against the oracle's
   dump at rows 1-4 in **absolute Pa**, bar **1.4e-4 Pa**. Free.
3. **The closed sub-basin's mass and sea-surface budget, rows 1-13.** The
   "budget closure before hypotheses" rule applied to the control volume nobody
   drew. If the barotropic volume budget closes and η still differs, the error is
   momentum; if it does not, everything above is downstream.
4. **Bottom form stress, inside the advection + kinetic-energy/pressure union.**
   With rows 1-4 blocked rather than periodic, the sidewall pressure term is back
   in the budget, and on a bathymetric staircase the depth-integrated pressure
   gradient differs from the gradient of the depth-integrated pressure by exactly
   the bottom form stress — wall-trapped, depth-integrated, geostrophic-scale,
   never isolated, and living in the same staircase construction that already
   carries 272.6 m of open transcription debt.
   *Discriminator*: NEMO gives the pressure gradient alone today; legoESM fuses
   it with the kinetic-energy gradient, so compare the unions and attribute the
   gap with NEMO's isolated term.

**Refuted by measurement, not argument.** Bottom drag — over-determined: one
site measured at seven orders below the bar, and all sites bounded together at
45% of the whole term. The implicit vertical solve's u-face control volume —
real (up to 272.6 m, exactly zero on flat faces) but anti-enriched at the wall
and under 1% against a 34% target; logged as transcription debt.

**Not returning to the top.** The barotropic free-surface solve as a whole. The
original pre-registration said a *refuted* friction hypothesis would send it
back there; friction was reclassified as a lever rather than refuted, and the
only piece of the solve still open (candidate 4) is far narrower than "the
solve" and is already fenced by three roundoff-level numbers.

**Also closed by measurement, not by argument.** The route by which the depth
mean of the friction reaches the state. legoESM folds it into the barotropic
forcing (`ocean_model_latlon_cgrid.py:3905-3911`) with the *same* thickness
weight the 3-D combine removes it with (`:8543-8548`), so nothing is lost or
double-counted by construction; and Part 3 measures it arriving at the wall rows
with slope 0.9987. The campaign's earlier `F_slow == zu_frc` to roundoff is the
same statement from the other side, since NEMO's `zu_frc` contains the
lateral-friction depth mean by construction (`dynspg_ts.F90:336-339`).

## Everything this does not establish

The raised-viscosity arms are not better physics; they are less faithful by
construction and two of the three fail a gate. Nothing here explains why the
band total is insensitive to viscosity while its meridional distribution is not,
beyond the telescoping argument, which is suggestive and not a proof. And the
compensation reading rests on three points of an anti-correlation with no
mechanism test behind it.

---

# Part 5 — the EEN vorticity flux, decomposed to the end; and two retractions

Written 2026-08-23. The ranked residual list above is now stale in three places
and is corrected here rather than left to be built on.

## The lead was right about the term and wrong about the mechanism

The EEN vorticity flux was the sole surviving candidate, at 3.10e-10 m/s² on
the four wall rows. It has now been decomposed exhaustively, and the
decomposition closes: NEMO's `vor_een` transcribed verbatim reproduces NEMO's
own dumped tendency to **2.0e-20 m/s²**, 7.7e-15 of the field's RMS. So the
ledger is complete, not merely ranked.

**The registered mechanism — the F-point thickness — is REFUTED as the owner,
and the reading behind it was right about everything except the size.** NEMO
applies *no* free-surface stretching to the F-point thickness at a vertex with
a dry neighbour (`fe3mask` is the four-`tmask` product, zero there), while
legoESM averages the live thickness over the wet count. The difference is real
and exactly where predicted — 1.47e-4 relative at wall vertices against 2.0e-6
at fully-wet ones, a 74× enrichment *in the e3f error itself*. Propagated
through the same triad code it delivers 3.25e-12 m/s², 95× too small, and
closes 0% of the mismatch. The campaign's standing lore that "the count-
normalised mean matches" was wrong in mechanism and right in consequence.

**Every input to the triad is exonerated.** F-point thickness, vertex Coriolis,
relative vorticity, face mass fluxes, the two face masks, the vertex mask, and
all of them together: each closes 0%. Two of them are not inert but do not
help — the boundary-vorticity convention is a 2.3× lever at the wall and 22.6×
in the far interior, confirming `nemo_live` is load-bearing, and the vertex
mask is provably unused under it.

**The owner is a missing metric weighting in the assembly.** NEMO weights the
meridional transport by the V-face zonal width and divides the assembled
u-tendency by the u-point zonal width (`dynvor.F90:791-792`, `:804`); legoESM's
Arakawa-Lamb triad used neither. Supplying it closes **96-98%** of the wall-row
disagreement and **86%** of the far interior — so it is a global fidelity
defect first measured at the wall, not a wall mechanism. NEMO's assembly driven
by legoESM's *own* inputs leaves only 2.1%, so the 12-point index pattern and
the triad-to-flux pairing were already identical.

It is also the difference between conserving the right quadratic invariant and
the wrong one. Measured on a closed domain with real metrics: with the
weighting the physical kinetic-energy norm conserves to 1e-16 and without it to
7e-10; without it what conserves instead is kinetic energy *per unit area*,
which is not an invariant on a stretched grid.

legoESM's barotropic solver has had this weighting all along
(`barotropic_coriolis="een_metric"`) and the DINO card already selects it, so
the card was running the metric-weighted EEN barotropically and the unweighted
one in the 3-D momentum. `pv_flux_ene` has the same gap.

## RETRACTION — the reduction every wall number was scored on

**Every wall-row number this campaign has published, including the 3.10e-10
headline, was called "signed thickness-weighted zonal mean". It is not.** The
reduction weights by the wet mask — an unweighted mean over levels — and DINO's
layers span 10.14 m to 545.20 m, so it over-weights the surface by ~54× relative
to mass.

| reduction | wall | far | enrich | the four wall rows |
|---|---|---|---|---|
| level mean (as published) | 3.098e-10 | 3.545e-11 | 8.74× | +5.49e-10 +4.48e-10 +2.17e-10 −2.50e-11 |
| **mass-weighted** | **2.500e-11** | 4.854e-12 | 5.15× | **−1.50e-11 −3.46e-11 −3.83e-11 +1.21e-11** |

The magnitude is 12.4× smaller and **the sign inverts on three of the four wall
rows** — in the depth mean legoESM's wall-row vorticity flux is more *eastward*
than NEMO's, not less. The reading that promoted this whole line of inquiry does
not survive its own instrument. **The drag and lateral-friction refutations were
scored on the same instrument and have not been re-run.**

## RETRACTION — the latitude argument

"The metric error grows as Δφ·tan φ and is therefore largest at the wall" is
false. On DINO's Mercator grid Δφ = Δλ·cos φ, so the growth cancels and the
ratio saturates: 1.008203 at the wall against 1.007706 at row 20, a 6% variation
that cannot produce an 8.74× enrichment. The enrichment comes from the
north-minus-south imbalance of the meridional flux, which is 6.98× enriched
because at the southernmost wet row the south face is land and the imbalance is
exactly one-sided.

## The 90-day A/B: the operator error does not carry the transport deficit

One controlled pair, same commit, differing in that one config field.

| | metric off (shipped card) | metric on | floor |
|---|---|---|---|
| wall rows 1-4 transport gap [Sv] | −0.2884 | −0.2946 | 0.091 |
| basin rows 0-13 [Sv] | −0.4258 | −0.4324 | 0.091 |
| sea-surface excess at the wall [mm] | 1.722 | 1.698 | — |
| circumpolar transport error [Sv] | 0.3823 | 0.3761 | 0.091 |
| upper density contrast [kg/m³] | 2.403e-4 | 2.399e-4 | 1.1e-4 |
| deep density contrast [kg/m³] | 1.429e-6 | 1.260e-6 | 4.5e-5 |
| southern surface σ max / mean [kg/m³] | 9.916e-5 / 2.936e-4 | 9.444e-5 / 2.977e-4 | 9.5e-5 |

All five gated metrics PASS at 5× on both arms, and **every single move is
inside its noise floor**. The wall-row transport changes by 0.0062 Sv, which is
**0.07× the floor**.

**So the registered falsification fired: the operator error is real, is now
closed to 2-4%, and is NOT what carries the −0.29 Sv wall-row transport gap.**
The wall-row tendency error and the wall-row transport error are different
objects. Nothing in the ranked list above survives as an explanation of the
basin deficit.

## Two ride-alongs, both closed

**The wall-row wind stress is exact.** Scored in absolute Pascals against a
1.4e-4 Pa bar — because an RMS-normalised score is structurally blind where the
stress is 0.23 mPa — the four wall rows agree to **2.2e-19 Pa**, fifteen orders
below the bar. The specific worry that the two models evaluate the profile at
different latitudes is void: on a latitude-longitude C-grid the u-points are
offset in longitude only, and the two latitudes agree exactly.

**The transport error is a recirculation, not a mass source.** Rows 1-13 are a
closed sub-basin (land at columns 0 and 51; the re-entrant channel is rows
14-48). If the 0.95 Sv deficit were a net convergence into that volume for a
year its sea surface would rise 24.9 m; the measured excess is 4.4 mm, so
**99.98% of the transport error recirculates inside the sub-basin**. Any
mechanism whose signature is net convergence is ruled out.

## What the next lane should do first

1. **Re-run the drag and lateral-friction refutations on the corrected,
   mass-weighted reduction.** They were decided on an instrument that
   over-weights the surface 54× and whose wall-row sign inverts. This is a
   rescore, not a run, and it is the cheapest thing on this list.
2. **Decide whether the DINO card should select the metric weighting.** It is
   NEMO's actual operator, it fixes the card's internal barotropic/baroclinic
   inconsistency, it conserves the physical energy norm, and it is neutral on
   every 90-day gate. It is left OFF pending that decision.
3. `pv_flux_ene` carries the same gap and is unfixed.

---

# Part 6 — the rescore, and the instrument lessons

## Every refutation, rescored on the corrected weighting

Offline, no runs. The rescore drives the same probes that produced the original
verdicts rather than re-deriving any term, and it is pinned to the
configuration the verdicts were taken under (metric weighting off), because the
card has since been flipped on and that changes the same term by two orders.

**A first version of this rescore was wrong, in the way the lane keeps being
wrong.** It declared lateral friction immune on the grounds that its probe
contains no call to the affected reduction — a test that counted a *name*. The
friction probe carries the same defect under a different name: its own row-mean
helper averages over longitude **and level** with no thickness weight, and one
of its outputs is an enrichment ratio, where reweighting moves numerator and
denominator differently. Friction is now actually rescored. Adversarial review
caught this; the immunity claim did not survive it.

| candidate | old verdict | rescored | |
|---|---|---|---|
| **bottom drag** | REFUTED | **REFUTED** — 3.93e-18 identical to the last bit under both weightings | **survives** |
| **friction, mask dimensionality** | REFUTED | **REFUTED** — exactly 0.0 under both | **survives** |
| **friction, e3 weighting** | REFUTED (1e-4 relative) | **REFUTED** — 6.75e-5 → 5.65e-5, a 16% move inside the same order | **survives on the leg that carried it** |
| **EEN vorticity flux** | CLEARS BOTH LEGS → sole candidate | **NO VERDICT under both weightings** | **the promotion was never supportable** |

**Nothing reopens.** Drag is immune by construction — its scored quantity is
already depth-averaged, so there is no vertical weight left to get wrong, and
that is measured (a 53× thickness contrast moves the 3-D reduction and leaves
the 2-D one bit-identical) rather than argued.

Friction's magnitude leg is what refuted it and that leg survives: the layer
weighting NEMO applies and the card does not changes the viscous tendency by
about 1e-4 *relative* on either reading, and a 1e-4 relative effect cannot
carry a 34% transport error whatever its spatial concentration. Its
**enrichment** leg does move, 2.35× → 5.16×, crossing the 3× bar — so on the
corrected weighting that difference genuinely *is* wall-concentrated. It is
simply far too small to matter, and it is reported rather than buried.

## The promotion was wrong twice over

The EEN vorticity flux was promoted to sole candidate on "clears both legs,
8.74× enriched against a 3× bar". Both halves fail:

- **The enrichment leg was scored on the wrong statistic.** The
  pre-registration defines enrichment on the row-mean *absolute* difference,
  and calibrated its 3× bar against comparators measured that way (2.4 for the
  viscosity thickness weighting, 0.57 for the implicit control volume). The
  8.74× came from dividing *signed coherent means* instead. On the registered
  statistic the term scores **1.86×** over the full interior and **2.81×** over
  the far interior — **it fails the enrichment leg on both interior sets**,
  under the published weighting, before any correction.
- **The magnitude leg then fails too** once the reduction is corrected:
  3.10e-10 → 2.50e-11 against a 5.9e-11 bar, with all four wall rows' signs
  inverted.

So the term that this lane spent its whole budget on never cleared the bar it
was said to clear. That is independently consistent with the 90-day A/B, where
closing 96-98% of it moved every gated metric by less than one noise floor —
two different instruments agreeing that it was never the owner.

## Three instrument lessons, and they belong together

The campaign's expensive errors have all been instrument errors, not physics
errors. These three are the same failure at three scales.

**1. Injection is not accumulation.** A per-step source and a drift are
different objects; a term can inject every step and accumulate nothing, and a
replay harness that drops a forcing manufactures the first while measuring the
second.

**2. A tendency error is not a transport error.** This lane's lesson, and the
sharpest one. The EEN vorticity flux disagreed with the oracle by 96-98% more
than any other operator this campaign has measured, at exactly the four rows
carrying the deficit, concentrated ~9× there. Closing it moved the 90-day
transport by **0.07 noise floors**. Operator fidelity and climate error are
different objects, and closing one is not evidence about the other — in either
direction. An operator can be wrong and inert; it can also be right and the
model still wrong.

**3a. A bar belongs to a STATISTIC, not to a quantity.** The magnitude leg and
the enrichment leg of the same shape test are registered on two different
reductions, and scoring both with one of them silently changed a 1.86× into an
8.74× — across the bar, in the direction that promoted a candidate. When a
pre-registration names a statistic, the code must read that statistic, and a
comparator calibrated on one is meaningless against the other.

**3. A reduction is part of the claim, and its NAME is a testable assertion.**
A statistic called "thickness-weighted" that averages over levels is not a
labelling slip: on this grid it over-weights the surface 54×, and it promoted a
candidate whose corrected sign is the opposite of the one that was reasoned
from. The helper was correct the whole time; every caller and its own docstring
were wrong. Before a reduction decides anything, print what it actually
computes on a case whose answer is known — a 53× thickness contrast that fails
to move a "thickness-weighted" mean is a one-line test that would have caught
this a year ago.

And **do not test for the defect by searching for its name.** The first attempt
to clear lateral friction counted occurrences of the broken helper in the
friction probe, found none, and declared immunity — while the same arithmetic
sat there inlined under a different name. A guard that can only fire on a
string it does not find is guaranteed to pass and proves nothing. Test the
behaviour: feed the reduction a known thickness contrast and require it to
respond.

The recirculation result belongs beside them as the positive form of the same
discipline: the closed sub-basin's budget says **99.98% of the transport error
recirculates internally** and only 4.4 mm of sea surface accumulates against
the 24.9 m a genuine convergence would have built. That is a control volume
drawn once, offline, that rules out an entire class of mechanism — every one
whose signature is net convergence — for the cost of an afternoon.

---

# Part 7 — the nonlinear terms, and the escalation

Every linear candidate was closed by Parts 5-6. The forcing matches to 2.2e-19
Pa, the deficit is a pure recirculation, and the surviving suspects were the
nonlinear momentum terms and their wall behaviour.

## What NEMO runs there, read from the compiled source

DINO sets vector-form momentum advection with the Hollingsworth kinetic-energy
scheme, so the horizontal advection is carried entirely by the KE gradient plus
the vertical advection, with the rotational part in the vorticity flux already
decomposed in Part 5.

**The Hollingsworth kinetic-energy scheme has a meridional stencil the standard
scheme does not.** Its cell-centre kinetic energy reads the rows above *and
below*, so at the first wet row it reads the land row — where the stored
velocity is exactly zero — and the cross term degenerates to a one-sided value
rather than a symmetric pair. legoESM builds the same stencil but fills the
off-row by edge replication rather than by reading a masked zero. Those two
agree **only because DINO's land row stores exactly zero**, which is a property
of the state and not of the code, so it is measured rather than assumed: the
probe plants a large velocity on that dry row and the kinetic-energy score moves
by **nineteen orders of magnitude**. The agreement is real and it rests entirely
on the dry-face gate.

The vertical advection has no meridional stencil for the zonal momentum at all —
its transport is a zonal average — so the wall rows are not special in its
construction. What *is* special is that its deepest wet level is a separate case
in which nothing is advected through the sea floor, and the wall rows are the
shallowest columns in the basin, so the fraction of the column governed by that
case is largest there.

## The three-way partition, and it is genuinely three-way

The residual list assumed the advection and the kinetic-energy/pressure terms
could only be compared as a union. They cannot only be compared that way: the
oracle dumps all three in isolation and legoESM's own helper returns its kinetic
energy and pressure gradients separately. So the partition is three-way, and the
union is kept only as a cross-check.

Three instrument controls passed before anything was scored. The oracle's own
partition closes **exactly** — its advection trend equals the sum of its two
isolated dumps to the last bit, on four files with four distinct checksums, so
the reader and the vector-form assumption are both right. legoESM's recomputed
pieces reproduce its own published diagnostics **bit-identically**. And the
stored velocity on every dry face is exactly zero.

The vertical-advection comparison is matched in quantity, which mattered: the
card selects the oracle-faithful advective form of the full velocity, not
legoESM's default perturbation form, so both sides advect the same thing.

## The three nonlinear terms match — but two of them could never have mattered

Scored on the three legs registered in advance, each on the statistic its own
bar was calibrated for. **The table now carries a headroom column, which the
first version lacked and which changes what two of the three rows mean.**

| term | wall difference | headroom | headroom / bar | largest single cell | verdict |
|---|---|---|---|---|---|
| kinetic-energy gradient | 4.3e-27 | 5.5e-11 | **0.93×** | 3.3e-24 | matches, but too small to matter |
| pressure gradient | 1.3e-19 | 1.5e-08 | **259×** | 4.8e-17 | **genuine agreement** |
| vertical advection | 1.0e-15 | 3.7e-11 | **0.62×** | 1.8e-14 | matches, but too small to matter |

Headroom is the score legoESM would earn by computing *zero* for the term — the
largest error it could possibly contribute at the wall. **For the kinetic-energy
gradient and the vertical advection the headroom is below the bar**, so those
two are refuted by their own size and the measured agreement is not doing the
work. Only the pressure gradient has real room to be wrong, and it agrees to
3e-12 relative. The meridional component of all three was also scored and
behaves the same way.

The instrument was checked from the oracle's side, which the first version could
not do: corrupting the reference — shifting it one row, one column, one level,
zeroing it, or pairing it with the wrong term — moves every score to at or above
the bar, while the true pairing sits ten to fifteen orders below.

**Retracted:** the premise that motivated this probe. legoESM's fill returns the
*true* neighbouring row for every scored row and differs from the oracle only on
rows that are entirely dry, so the two stencils agree for any land value — not
"only because the stored velocity there is zero". The mechanism did not exist.
The measurement stands on its own.

## RETRACTED: "candidates exhausted" — the term list was incomplete

The first draft of this section escalated the campaign on the grounds that every
operator had been compared. **That was wrong, and adversarial review overturned
it.** The partition covers only the terms inside the tendency calculation.
Roughly a third of what reaches the wall-row velocity in this configuration
happens *after* it, and none of it has ever been compared against the oracle:

- the leap-frog recombination of the split velocity;
- the split-explicit barotropic solve (its own gate sits eight times outside its
  bar, and its note says the residual is diffuse and unattributed);
- **the implicit vertical solve, whose gate row is blank and whose only
  end-to-end number is a 15% discrepancy — the largest unexplained figure
  anywhere in the campaign, on the one stage never isolated**;
- the after-level reconciliation, which *sets the depth-uniform part of the
  velocity* while the deficit is itself depth-uniform, and whose gate row is
  also blank;
- the time filter's composition with all of the above, and the free-surface
  drift correction that has no oracle analogue.

**And a real defect was measured during that review.** One step uses **two
different column depths for the same velocity**: the barotropic/baroclinic split
uses the masked minimum rule, while the implicit vertical solve divides by an
*unmasked* face average that keeps half a cell of rock at every topographic
step. The divisor is 0.3–1.2% too deep, and the error carries a ~0.9
percentage-point meridional gradient concentrated across the first twenty rows —
the exact band the deficit occupies. So the step removes the barotropic mean
with one rule and re-adds it with another. This was a known open item filed as
"not measured"; it is measured now.

**The second thing review overturned:** refuting a term by its largest single
cell is sound only against *direct* forcing. At a bridged state the two models
are identical, so the probe measures the forcing difference alone and says
nothing about how a difference grows once the trajectories separate. This
campaign has already proved that map is badly non-linear in the other direction
— closing 96-98% of a genuine operator error moved the transport by 0.07 noise
floors — and a map that non-linear cannot be inverted to license "small now,
therefore never the owner".

## The claim that IS supported, and what to do next

> No term of the baroclinic momentum tendency differs from the oracle by enough
> to force the wall-row deficit **directly** at the day-180 state. Rectification
> through the trajectory is untested, and the terms outside the tendency
> function are unexamined.

Four operator-level tests remain, all cheap, none requiring a new oracle run:

1. **Fix and re-measure the column-divisor mismatch.** The only *confirmed*
   defect on the list; the mask it needs is already computed three lines away.
2. **Diff one committed step against the oracle's own next restart**, bisected
   across the four post-tendency stages using dumps already on disk. Highest
   information available for zero new oracle work.
3. **The after-level reconciliation against its own bracket** — it sets the
   depth-uniform velocity, and the deficit is depth-uniform.
4. **The implicit vertical solve in isolation**, which needs a port rather than
   a bracket, but whose 15% discrepancy should not stay unexplained while
   cheaper tests go unrun.

Trajectory-feedback experiments are the tier *after* those, and by then they
would be justified by evidence rather than by exhaustion.

## Two pieces of debt this lane uncovered

**The Hollingsworth stencil's boundary fill is not the oracle's convention.**
Zero-Dirichlet gives one cross term; edge replication gives roughly four times
that for smooth flow. They coincide here only because the boundary row is dry
and stores zero. On a restart from a model that does not zero land velocities,
or a wetting-drying case, they would differ by a factor of four at any wet
boundary row.

**And that stencil is not halo-aware.** It fills the neighbouring row with a
bare edge replication and no halo exchange — verified, there is none before the
call. Under latitude-decomposed parallelism a rank whose first row is an
interior row would silently replicate its own data instead of reading its
neighbour's. Filed as debt; not fixed here.
