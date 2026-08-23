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

## Part 4 — the two ranked discriminators: drag refuted, the vorticity flux is the lead

Pre-registration: `PREREG_wall_drag_and_een.md`, written before either number
existed. Probe: `wall_term_discriminators.py`. Both read on NEMO's own one-step
single-rank dump (kt 5761) from the restart the twins start from, with legoESM
bridged from that same restart on the same vertical ladder — bit-identical state
on both sides, so a difference is the term and not the trajectory.

The registered shape test, the same for both: to carry the defect a term's
difference must be **both** wall-enriched (≥3× the interior) **and** large enough
(≥5.9e-11 m/s², which is 4.6e-4 m/s divided by 90 days — what it would take to
build the measured velocity error if the difference accumulated coherently for
the whole window, a deliberately generous upper bound).

### Bottom drag — REFUTED, on all three of its parts

| | measured |
|---|---|
| the coefficient at u-faces | **bit-exact**: 0 of 9758 cells above 1e-15 |
| the bottom-level index vs NEMO's | **0 of 9758 disagree**, 0 of them in the wall rows |
| the assembled increment to the barotropic forcing | wall rows **4.4e-15 m/s²**, interior 3.8e-14, enrichment **0.12×** |

Anti-enriched at the wall, and three orders below even the *refute* bar of
5.9e-12. Robust to how the interior is defined (0.12× or 0.23×). **The
top-ranked candidate is refuted.**

The bottom-level index result is worth calling out on its own: an off-by-one
there would have been wall-concentrated by construction and would have made
everything downstream meaningless. It is not there.

*Scope, stated rather than glossed*: this measures **one** of drag's three
composition sites — the baroclinic-residual fold into the barotropic forcing.
The in-matrix implicit term and the explicit bottom-stress source at the bottom
cell are not covered. The bit-exact coefficient and the exact bottom index
constrain both, but do not close them.

### The EEN vorticity flux — clears magnitude by 11×, shape undecided

| | measured |
|---|---|
| wall-row difference | **6.49e-10 m/s²** — 11× the 5.9e-11 pass bar |
| interior (rows 8–150) | 3.00e-10 → enrichment **2.17×** |
| far interior (rows 20–150) | 1.85e-10 → enrichment **3.51×** |
| relative agreement | 1.4e-5 on a term of 4.7e-5 m/s² |

**The enrichment leg is undecided by the data, and that is a defect in my
pre-registration, not a result.** I registered a 3× enrichment bar without
pinning which rows count as interior. The parent document measured the wall gap
decaying over about **eight** rows, so rows 8 and 12 sit *inside* the boundary
layer and dilute the denominator; excluding them takes the ratio from 2.17× to
3.51×, across the bar. Both readings are reported and neither is presented as
the answer. I noticed this after seeing the numbers, which is exactly why it is
recorded as undecided rather than used to re-cut the verdict.

**Registered outcome: neither candidate promoted.** But the vorticity flux is
the largest term difference this campaign has measured at the wall — **five
orders above bottom drag** — and the first term found that is even capable of
producing the deficit. It is the lead, labelled **PLAUSIBLE**, and the next step
is a sharper localisation (fit the difference's own decay profile against the
gap's, rather than comparing two hand-picked row sets) — **not** a build. The
pre-registration's rule stands: offline shape match before any fix.

### An inconsistency the self-test surfaced

The two wall conditions in this model are **deliberately different from each
other**, and that is faithful. DINO sets `ln_dynvor_msk = .false.`
(`namelist_cfg:333`), so the coastal f-point is **not** masked out of the
vorticity flux: the relative vorticity at the wall corner is the shear between
the first wet row and the land row's stored zero — a **no-slip-like** shear.
The *same* model's lateral viscosity treats that same corner as **free-slip**
(`fmask = 0`, Part 1 row 5). legoESM matches both, by `een_q_boundary="nemo_live"`
and by the corner mask respectively.

So the wall vorticity in both models rests on the land row's stored value being
exactly zero. The probe measures that it is (max |u| on dry faces = 0.000e+00),
which is what makes the two models agree there at all — and it is why the
self-test reports that read rather than forbidding it.

---

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

**Cause unknown, and the list is shorter than it was.** Two of the four
candidates below have now been measured. Ranked by what the measurements left
standing.

1. **The EEN vorticity flux at the wall — the lead.** The only term whose
   wall-row difference clears the magnitude bar, by 11×, and five orders above
   bottom drag. Its enrichment straddles the bar depending on where "interior"
   starts (2.17× or 3.51×), which the pre-registration failed to pin.
   *Cheapest next step*: fit the difference's own meridional decay against the
   gap's measured 8-row decay, rather than comparing two hand-picked row sets —
   a stronger shape statistic on data already in hand, offline, no run. Only if
   the profiles match does a fix get designed.
2. **The other two bottom-drag composition sites** — the in-matrix implicit term
   and the explicit bottom-stress source at the bottom cell. The fold into the
   barotropic forcing is refuted and the coefficient and bottom index are exact,
   which constrains these two hard but does not close them.
   *Cheapest discriminator*: the same probe pointed at the `dynzdf` dumps.
3. **Momentum advection and the kinetic-energy / pressure-gradient pair.**
   Separable today: compare the unions `{KEG+ZAD+HPG}` against
   `{KE_PGF + vertadv}`, with the vorticity and friction groups each alone.
   *Cheapest discriminator*: that union comparison on the same one-step state
   this Part-4 probe already loads.
4. **The barotropic solve's face depth at the last wet row.** Still last, behind
   three roundoff-level numbers (substep 1 bit-identical at 3.1e-15, the
   sea-surface operator at 9.3e-15, the transport reconstruction at 7.3e-16).

**Refuted by measurement, not argument.** Bottom drag's fold into the barotropic
forcing (Part 4). The implicit vertical solve's u-face control volume — real, up
to 272.6 m, exactly zero on flat faces, but 0.48% at the wall against 0.84%
interior, so anti-enriched and under 1% against a 34% target; logged as
transcription debt.

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
