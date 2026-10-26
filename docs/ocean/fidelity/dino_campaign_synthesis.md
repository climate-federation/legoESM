# The DINO / NEMO fidelity campaign — synthesis and hand-off

**Audience:** someone picking this up cold. No prior thread required.
**Scope:** the state of the legoESM ocean twin of the NEMO DINO configuration as
of 2026-08-26, what is settled, what is not, and what to do next.
**Status of this document:** consolidation only. Every number below is carried
from a committed source (issue #1455's comment record, a merged PR, or a result
document in this directory) and cited in place. Nothing here was re-derived or
re-measured. Where two sources disagree, the later retraction wins and the
supersession is stated.

Tracker: issue **#1455**. Result documents: `dino_verdict360_result.md`,
`dino_basin_budget_result.md`, `dino_wall_ldf_alignment.md`,
`dino_wall_fixed_bias.md`, `dino_wall_flicker_source.md`,
`dino_eta_wave_field_result.md`. Debt register: `dino_outstanding_fidelity_debt.md`.

---

## 1. The result

### 1.1 The registered headline, verbatim

From `dino_verdict360_result.md`, lines 3–9. (The *pre-registration* is a
separate file, `PREREG_verdict360.md`, committed before any run started; these
lines are the result it was registered against.)

> **Result, in one sentence.** From a shared NEMO day-180 restart, one year of
> legoESM drift away from NEMO is **no larger than the two models' own
> run-to-run spread in the circumpolar channel**, and **larger than it in the
> southern basin**. This is a statement about one year from a common ocean
> state; it is not a statement that the two models share a climate, and the deep
> ocean has not adjusted on this timescale.

The single sharpest number, from the same document:

> **The channel-band gap over a full year is 0.0006 Sv on a 34 Sv transport —
> 17 parts per million.**

That statistic is labelled **post-hoc** in its own source (it averages all 19
scored days to cover one complete seasonal cycle, rather than reading one
endpoint carrying one season's phase). The **pre-registered** endpoint result is
the day-360 table below.

### 1.2 The verdict table — day 360

Both models integrated 360 days from the same NEMO restart
(`DINO_00005760_restart.nc`, day 180 of the DINO spin-up) at 2700 s. legoESM ran
the shipped `nemo_dino_kamm_mlf` card with no option flags; NEMO ran the
certified binary on the recorded 90-day twin's namelist with only the run length
changed. Each side additionally ran three members perturbed by a 1e-14 relative
temperature nudge, and the spread among those members is the floor a real
difference must clear.

The rule, fixed before any number existed: a metric is **INDISTINGUISHABLE**
when `|gap| ≤ 2 × floor`, with `floor = sqrt(legoESM_spread² + NEMO_spread²)` at
the same day.

| metric | gap [Sv] | floor [Sv] | gap/floor | verdict |
|---|---:|---:|---:|:--|
| channel band (e3t_0, mean) | **+0.0015** | 0.0101 | **0.15** | **INDISTINGUISHABLE** |
| channel band (group reduction) | **+0.0024** | 0.0096 | **0.25** | **INDISTINGUISHABLE** |
| channel band (e3t_1d, median) | +0.0396 | 0.0126 | 3.15 | no — upper bound only |
| north of band | −0.0385 | 0.0041 | 9.32 | no — upper bound only |
| **south of band** | **−0.9519** | 0.0617 | **15.42** | **no** |
| full section (mean) | −0.9880 | 0.0558 | 17.71 | no |
| ACC (the recorded gate metric) | −0.9704 | 0.0454 | 21.37 | no |

Source: `dino_verdict360_result.md`, "The verdict table — day 360". Probe
`verdict360.py`; pre-registration `PREREG_verdict360.md`; result commit
`a1387f1f7`.

**The full-section gap is the southern-basin gap.** Split by latitude at day
360 the three bands sum exactly to the **mean-reduced** full section: south
−0.952, channel +0.002, north −0.038, total −0.988 Sv. (The separate "ACC" row,
−0.9704, is a *median* over longitudes and is therefore **not** additive over the
bands — do not try to decompose that one.) The model is right in the channel and
wrong in one basin, and a single all-latitude number cannot say that.

### 1.3 The tolerance gate: 4 of 5, and what the gate is

Separately from the verdict run, the campaign carries a five-metric **acceptance
gate** at 90 days (ACC transport, upper density contrast, deep density contrast,
southern-band σ maximum, southern-band σ mean). Its thresholds are fixed
constants and are never modified to pass a run.

The corrected-clock baseline (issue #1455, "NEW BASELINE (corrected clock, day
90)") was the campaign's first valid table: **4 of 5 PASS, ACC the sole FAIL**.

| metric | signed gap | ×floor | verdict |
|---|---|---|---|
| ACC | **+1.872 Sv** | +20.6 | **FAIL** |
| upper contrast | +4.0e-04 | +3.6 | PASS |
| deep contrast | +6.1e-05 | +1.4 | PASS |
| σ surface max | −3.6e-04 | −3.8 | PASS |
| σ surface mean | −2.9e-04 | −3.1 | PASS |

Gate constants unmodified; day-0 bit-identical; harness change `8e56daafa`;
baseline `3855a2173`.

**One unresolved conflict on that baseline, recorded because the record never
closed it:** the ordered day-180 walk's shipped-grid arm reads **+1.659 Sv**
where this recorded baseline reads **+1.872**. Same bit-deterministic harness, no
model change. The source calls the discrepancy *"real and unexplained"* and it
was never resolved. Treat the +1.872 as the recorded value, not a verified one.

A **5/5 pass was subsequently reached and shipped** — the NEMO-faithful
barotropic reconciliation pair became the `kamm_mlf` card default (`1d5b19a1d`),
measured 5/5 at **0.411 Sv** ACC gap on the 90-day gate *at the moment of
shipping*. That figure has since moved twice and should be date-stamped whenever
it is quoted: the three drag fixes took the day-90 channel gap from 0.41 to
**0.382 Sv**, which is what the shipped card reads today.

Note also what the **−0.401 Sv full-section / +0.063 Sv channel** pair is and is
not. It is a **re-measurement, not a later improvement**: the source states that
the older −0.597/+0.287 figures *predate* the faithful reconciliation pair, and
that on the shipped card the same reductions give −0.401/+0.063 — i.e. the pair
had **already** cut the channel gap 78% when those older numbers were being
quoted. Do not read it as "later fixes moved the metric".

So "4/5" is the honest description of the corrected-clock baseline, and "5/5" the
honest description of the shipped card; both are tolerance verdicts, not
noise-floor verdicts — see next.

### 1.4 What "indistinguishable" does and does not claim

Three limits, each from its own source, and each one a thing a cold reader will
otherwise get wrong.

**Horizon.** The claim covers **one year from a common ocean state**. It is not
a claim that the two models share a climate. The deep ocean has not adjusted on
this timescale. (`dino_verdict360_result.md`, the headline paragraph itself.)

**Tolerance is not noise floor.** The gate's five constants are **tolerances
with provenance**, not measured noise floors, and the campaign's own
recalibration says so in as many words. Measured directly on a 4-member
perturbed ensemble at 90 days (branch `fidelity/dino-floor90`):

| metric | constant in use (from a 10-yr ensemble) | measured 90-day spread | ratio |
|---|---|---|---|
| ACC transport | 0.091 Sv | **1.15e-05 Sv** | 1/7,900 |
| upper contrast | 1.1e-4 | 5.0e-08 | 1/2,200 |
| deep contrast | 4.5e-5 | 2.9e-09 | 1/15,500 |
| σ max / σ mean | 9.5e-5 | ~1.5e-07 | 1/620 |

The 10-year ensemble that set the constants was fully saturated by years 3–5; a
1e-14 perturbation grows about one order per 30 days and reaches only ~3e-4
relative by day 90. Consequence, stated plainly in the source: **"5/5 at the
gate" means "within declared acceptance tolerances", NOT "indistinguishable from
NEMO."** Re-scored at the measured floor, the shipped arm's 90-day transport gap
is **~25,000 floors** — and the campaign's later full-section budget scores the
same card at **25,000× (full section) and 3,900× (channel)**. The gate constants
were **not** touched — changing them goes through the owner — but they should
lose the "noise floor" label.

**Read that ratio's denominator carefully.** Those multiples are quoted verbatim
and are scored against the *measured 90-day **difference** floor* — the bar for
comparing two runs — which is **not** the single-run spread tabulated above.
Dividing a gap by 1.15e-05 will not reproduce them and is not how the source
computed them. Quote the multiples, or quote the spread; do not mix the two.

The NEMO side was then measured too: NEMO's own 90-day spread is **1.5e-07 Sv**
against legoESM's 1.15e-05, i.e. 10–100× tighter, and the combined RSS bar
changes every metric by under 0.1%. The recalibration therefore stands
two-sided. legoESM's larger spread is an **upper bound** — fp32 snapshot storage
bounds it from above.

The one place the 0.05–0.09 Sv constant is validated is the **1-year** horizon:
prediction P1 in the verdict run confirmed it, for the **ACC transport** metric
specifically (legoESM day-360 spread 0.0447 std / 0.102 range against the
transferred 0.050 / 0.091 — growth from day 90 of ×2200). Two things follow, and
they are different claims: P1 validated **one constant's magnitude class** at one
year, and separately **every floor in the day-360 table above was measured
directly** from that run's own ensembles rather than transferred. The 90-day
gate's constants are the transferred ones.

**Deep ocean.** Excluded explicitly by the headline sentence. Density metrics at
day 360 all score `no` against floors of 1e-6 or less, with absolute gaps 1e-4
to 6e-4 kg/m³ (verdict run, P4, which carried no prediction).

**The five limitations the verdict document records against itself**, all from
`dino_verdict360_result.md`, "Limitations a reader should carry away". These are
not hedges; two of them change how the headline should be read.

1. **The floor is effectively one model's dispersion.** NEMO is 9× to 16,600×
   tighter than legoESM under the identical nudge, so **77–99% of the combined
   floor is legoESM's own spread**. The pre-registration justified combining the
   two as "√2 × one side when the two wobble equally"; that justification is
   **empirically false here**, and the correction is disclosure, not a different
   formula.
2. **One year is not a climate.** Both models still carry the same deep ocean
   from the shared restart, and the overturning adjusts on decades.
3. **Seasonal aliasing.** Each registered horizon carries a seasonal phase; the
   full-cycle mean is the remedy and it is post-hoc.
4. **n = 4.** Every floor carries about 41% relative standard error and is a
   factor-of-two estimate.
5. **"ACC [Sv]" is a reference-geometry proxy** (e3t_1d weighting, which differs
   from the model's own thickness by up to 12.9% on wet cells). It is the
   identical functional on both sides, so the **gap** is valid; the **number**
   must not be quoted against a published or observed ACC.

One further item is recorded as open rather than as a limitation: legoESM
amplifies an identical 1e-14 nudge **78–435× more than NEMO** by day 90, with
the offset born in the first 30 days rather than in the growth rate. Not
investigated.

**A caution about the transport metric alone.** Raising lateral viscosity
"improves" the channel transport from 0.382 to 0.069 Sv while the density
contrast behind it degrades monotonically and fails its own gate from 1.5×
(`dino_wall_ldf_alignment.md`; issue #1455, "Wall operator: MATCH on every
row"). The transport metric is tunable by a knob that breaks the physics, and
the density gates are what catch it. Score them jointly; never transport alone.

---

## 2. The open problem

### 2.1 Statement

**The southern basin runs −0.95 Sv weak at one year and compounds.** It is the
campaign's single named work item, assigned by a rule registered before the
numbers existed (bins: `≤2` floors closed at one year, `2–10` on the board,
`>10` the next single work item; measured 15.4 endpoint / 20.9 registered
90-day window / 22.9 full year — all three agree).

**On the compounding factor.** Issue #1455 quotes "compounds ×2.2/yr". That
figure appears **only in the issue comment** — a full read of
`dino_verdict360_result.md` does not contain it — and the document it summarises
qualifies the reading directly: the trajectory is +0.03 (day 10), −0.43 (day 90), −0.23
(day 180), **−0.06 (day 270)**, −0.23, −0.60, −0.95 (day 360). The source says
in as many words that *"Reading only the two endpoints would say 'it doubles';
the full trajectory says it compounds with a mid-year recovery, which is a
different and more specific fact."* Use the trajectory, not the ratio.

**And the day-270 near-zero is not agreement.** A later measurement corrects it.
Splitting the basin transport into a bottom-referenced part and the rest, the two
models at day 270 disagree by about **1.74 Sv one way and 1.56 Sv the other**;
the net is small only because those nearly cancel — a **19-fold cancellation**
on the four-member mean (net −0.18 Sv), and **53-fold on the single control
member** (net −0.06 Sv, which is the value the verdict table reports). In the
source's words: *"Reporting day 270 as 'momentarily indistinguishable' is true of
the metric and false of the physics."* The earlier reading is retracted by its
own author. Note the two reductions are not the same number — the verdict table's
−0.06 is the control member, the ensemble's is −0.18.

### 2.2 What the deficit physically is

From `dino_basin_budget_result.md` (phases 1+2, dual-reviewed, `660942b08`):

- One **depth-uniform velocity difference of −4.6e-4 m/s** — about half a
  millimetre per second, at every level — in a roughly **300 km strip against
  the southern wall**. That single constant reproduces **75.7%** of the day-360
  per-level difference by variance, and predicts the measured −92 m shift in the
  transport's depth (predicted −107 m). The competing "same flow, weaker"
  description explains 37.5% by variance and predicts no shift at all.

  *Internal conflict, flagged rather than reconciled:* the document's own
  headline says this null explains "**93%** of its total" while its later table
  gives **65.5%** by total (residual −0.070 of −0.935 Sv). The variance figures
  agree (76% vs 75.7%); the totals do not. **The later number in the file is
  65.5%** and the text does not reconcile them. Use the variance figure, which
  is unambiguous.

- "Uniform" describes the bulk of the column, not the bottom cell. In the bottom
  cell specifically — the level that sets the bottom-drag rate — the two models
  differ by **−1.5e-3 m/s** at day 360, about three times the uniform value
  fitted on the same day.
- **Four rows (69.5–68.4°S) carry −0.905 of the −0.952 Sv.** The row shape is
  stable all year (0.96 correlation, day 90 vs day 360).
- legoESM builds circulation in those rows **23% slower, sustained** over the
  year-long momentum budget (rate +0.3750 against NEMO's +0.4866 m³/s² per row,
  year-mean shortfall **0.112**; instrumented oracle bit-identical at both day 90
  and day 360). Caveats travelling with it, both from the source: across the 36
  ten-day windows the spread is **0.322 — three times its own mean — and 13 of
  36 windows have the opposite sign**, so *"no per-window or per-season
  attribution is supportable from this budget. Only the year mean is signal."*
  And each model's own budget closing to four decimals is an **instrument**
  check; once both sides close, the cross-model step is arithmetic, not
  independent evidence about mechanism.
- It is a **recirculation, not a convergence**. Had the missing transport been
  converging in the closed sub-basin for a year its sea level would stand 24.9 m
  high; it stands 4.4 mm. **99.98% circulates internally** — every
  net-convergence mechanism is ruled out.
- The transport error and the surface excess are **one object**: 0.25 mm per row
  predicts 1.0 mm of the measured 1.72 mm over four rows.

- **The shape is an overshoot, not a shortfall, and this is the signature every
  candidate has to reproduce.** legoESM's westward wall return lobe is **34% too
  STRONG** while the rest of the basin is right to 3%. In the source's words it
  *"is not a failure to build the eastward flow — it is an overshoot of the
  wall's return lobe."* Getting this direction backwards sends the next lane
  looking for a missing source instead of an excess one.
- The balance is **geostrophic**: the day-360 circulation difference decomposes
  into **91.0% sea-surface slope, 6.7% density gradient, 2.3% ageostrophic**
  (robust 78.9–94.1% across aggregates, and 93.3–95.6% across four member
  pairs). Registered verdict: CONFIRM geostrophic. Correctly flagged in its own
  source: geostrophy here is **diagnostic** — any wall-trapped momentum error
  adjusts to it — so the registered *consequence* does not follow. It buys a
  target and an elimination, not an owner.

### 2.3 The negative-space map — everything exonerated, with its decisive number

This is the most valuable part of the hand-off. Each row is a candidate that was
tested and eliminated, so that nobody spends a week re-eliminating it.

| candidate | verdict | the decisive number | where |
|---|---|---|---|
| **Surface forcing / wind** | EXONERATED | the oracle's dumped wind stress matches the analytic form legoESM uses to **1e-10 per row**, and the torque **legoESM** applies to these rows is **17.811 in every one of the 36 ten-day windows** — DINO's wind has no time dependence at all. (One-sided constancy: the oracle's vertical-solve row reads 18.100, explained in the source.) **Scope:** this exonerates the wind *forcing*. The wind-stress **placement** at a wind-driven wall row is a separate question and is still OPEN — see §2.4 item 3. Earlier, independently: identical analytic forcing, agreement 6e-17. Wall-row wind stress exonerated to 15 decimals in absolute units. | `dino_basin_budget_result.md`; issue #1455 "§D consolidated update", "The EEN chase" |
| **Water masses** | EXONERATED as the *bulk* cause; the gradient is a small same-signed contributor | basin-mean densities agree to **4e-5 kg/m³** on a 1027 kg/m³ field, so mechanisms working by changing the water masses are ruled out **at the basin scale**. The document's own headline adds a caveat — that the density *gradient* in the rows owning the gap was never measured — and **that caveat is superseded later in the same file**: the gradient WAS measured, with a common reference profile so the hydrostatic common mode cancels rather than being differenced, and it comes out at **6.7% and same-signed**, i.e. it *adds to* the explanation rather than competing with it. Three independent implementations agree. The later measurement wins. | `dino_basin_budget_result.md` (headline vs Part 4) |
| **Bottom drag (as a defect)** | REFUTED, over-determined | drag coefficient bit-exact (0 of 9758 cells differing), bottom index 0/9758, assembled increment **3.9e-18** — seven orders under the bar. The whole drag term bounds any drag-borne defect at 45% of itself, impossible against a bit-exact coefficient. Independently: 0.1% collapse under substitution against a pre-registered bar. **Supersession:** `dino_basin_budget_result.md` records drag as "not shaped like the deficit, and not retired"; the later bit-exactness result refutes it. Later wins. **Rescore status:** the *earlier* drag refutation was decided on the layer-averaged instrument later retracted (see §4, layer-vs-thickness weighting), and its rescore is recorded as outstanding. The bit-exact coefficient and the 0.1% substitution collapse are independent of that instrument and are what carry this verdict — the rescore would confirm, not decide it. | issue #1455 "The funnel converges"; `dino_wall_fixed_bias.md` |
| **Lateral friction / free-slip wall treatment** | **EXONERATED end-to-end — RESTORED 2026-08-27; the PROVISIONAL downgrade is RETRACTED** | free-slip identical at **0 disagreements out of 372,528 corner points, on every one of the 36 levels**; wall stress measured **exactly 0.0 on both sides**. The 13-row alignment table finds no wall-treatment difference that could carry the lobe: 11 rows MATCH outright, and the two rows carrying a real code-level difference quantify to **exactly 0.0** on the actual state (a mask-dimensionality difference) and **1e-4 relative** (a thickness-weighting difference). Viscosity ablation at 1.0/1.5/2.0× (rn_Uv 0.27/0.405/0.54) gives unsigned southern-band error **0.4459 → 1.1407 → 1.8002 Sv**, linear to **1.6%** and monotone — *"the lateral viscosity is a LEVER on the wall error, not its OWNER."* **RESCORE STATUS — SETTLED, and the previous entry here was wrong twice.** (1) The rescore was not outstanding: it landed on **2026-08-23** in `d731e8154`, three days BEFORE this document downgraded the row, and is carried in Part 6 of the source file. This row's earlier text quoted that file's Part 5 *"what the next lane should do first"* to-do list without reading the Part 6 that had already done it — a pointer read as a fact. (2) Its supporting claim that friction had *"no independent later evidence"* was also false: the viscosity ablation is scored on **90-day Sv transports**, a statistic with no vertical weighting in it at all, so the retracted reduction cannot touch it. **Rescored numbers, re-run at HEAD on 2026-08-27** (`wall_reduction_rescore.py`, self-test PASS): mask dimensionality **exactly 0.0 under both weightings**; the e3-weighting leg **6.7541e-05 → 5.6500e-05**, a 16% move inside the same order — and a ~1e-4 relative effect cannot carry a 34% transport error whatever its concentration, so the leg that carried the refutation survives. **One leg genuinely moves and is reported rather than buried:** the ENRICHMENT leg goes **2.35× → 5.16×**, crossing its 3× bar, so on the corrected weighting this difference really is wall-concentrated. It is simply far too small to matter. | `dino_wall_ldf_alignment.md` Parts 2, 5 and 6; issue #1455 "Wall operator: MATCH on every row"; rescore re-run 2026-08-27 |
| **EEN vorticity scheme** | FIXED and INERT | before the fix it was reported as the **worst-matched operator of the campaign** — pointwise relative disagreement 2.1e-3 at the wall against a matched-operator floor of 1.5e-5, wall reduction 5.25× the bar, enrichment 8.74×. **Those pre-fix figures are layer-averaged and therefore come from the instrument §4 records as retracted**; mass-weighted, the enrichment is **5.15×, not 8.74×**. Quote the mass-weighted value. The *fix* below is unaffected — it was decided on a matched-state substitution, not on the enrichment. The true owner of the tendency mismatch was then found: NEMO width-weights the north-south transport in the vorticity scheme and divides by the local width; legoESM did not. Supplying it closes **96–98%** of the wall-row disagreement. NEMO's `vor_een` transcribed verbatim reproduces NEMO's own dumped tendency to **2.0e-20 m/s², i.e. 7.7e-15 of the field's RMS** (issue #1455 renders this as "15 significant figures"; the document's own number is the one quoted here). The controlled 90-day pair then moved **nothing**: wall rows −0.2884 → −0.2946 Sv, basin −0.4258 → −0.4324, circumpolar 0.3823 → 0.3761, all inside floors, all five gated metrics PASS on both arms. Switched ON for the oracle card anyway on faithfulness grounds. | `dino_wall_ldf_alignment.md` (all figures); issue #1455 "The EEN chase" (rounded restatement) |
| **e3f construction at dry-neighbour vertices** | REFUTED as owner | real — a 74× enrichment in the e3f error itself — but propagated through the triad it delivers only 3.25e-12 m/s², **95× too small**, closing **0%** of the mismatch | `dino_wall_ldf_alignment.md`; issue #1455 "The EEN chase" |
| **Face-depth divisor** | FIXED and INERT | three-arm A/B: climate-inert to **0.0000 Sv** on the wall band; refuted as owner against a 0.05 Sv registered bar. The handed premise was also refuted — the divisor error is *smallest* at the wall (0.57×). Shipped on correctness; the repo now carries one face-depth rule. | issue #1455 "Post-tendency bisection" |
| **Every post-tendency stage** | REFUTED — wrong sign AND wrong shape | one-step stage table from an identical day-180 state: barotropic solve + recombination −0.047 Sv, after-level reconciliation +0.047 (one object, 99% cancelling by design), implicit vertical solve +1e-5, filter 0. Committed net **+4.9e-4 Sv with the wrong sign and 9× interior-enriched**, against a registered requirement of wall-enriched. Magnitude leg honestly recorded INCONCLUSIVE; the refutation rests on sign and shape. **Scope the source attaches:** refuted *"under linear retention only … not that these stages can never matter."* | issue #1455 "Post-tendency bisection" |
| **Convective adjustment's hard switch** | SHARED, not differential | the switch is a genuine ~300× rectifier (edge sharpness alone 294×, trigger location 6.9×) but **both models have it** — NEMO's own EVD is also a hard switch. Per unit of perturbation legoESM's switch fires **less** (0.57–0.61; NEMO is 1.6–1.75× more trigger-happy) and the two models' stratification is equally tippable (≤1.045×). The raw 4.8–8.6× flip excess is a growth-difference confound. | issue #1455 "The rectifier is NAMED", then "Grown-noise census" |
| **Per-step injections (barotropic, both survivors)** | EXONERATED for the accumulation | the retention operator was measured directly across eight 90-day arms over a 200× amplitude range: a deposit-shaped one-step kick retains R(day 1) ≈ 0.19–0.29 and falls below 1% by day 5–7. Retention-corrected, the frozen-forcing share integrates to +0.049 Sv and the in-loop constant to +0.037 Sv against a needed −0.401: **8–11× short, wrong sign.** | issue #1455 "Retention verdict" |
| **Meridional (sidewall) 2Δt flicker** | TRANSIENT, cancels | 6.6× early → **0.95× late**, locus collapsing 24× → 3.9×, and **depleted (0.90) in member differences** — it cancels in exactly the statistic ensemble spreads are made of. Nothing downstream. | `dino_wall_flicker_source.md`; issue #1455 "Flicker hunt FINAL" |
| **Zonal (±69.5° end-wall) excess** | SUSTAINED, **still unowned** | **2.68 / 2.85 / 7.59** under the three estimators, late-half, with CI [1.26, 5.10] excluding one, both floors cleared, second start state reproduces, member-difference enrichment 8.6×, damping equal within ~1.5×. Later reframed: **95% of it is a FIXED field**, which every two-step instrument cancels by construction. | `dino_wall_flicker_source.md`; issue #1455 "Re-projection verdict" |
| **Grid metric (v-face zonal width)** | **PARTIAL — ~25% at the walls; the CONSTRUCTION DEFECT is now FIXED (2026-08-27)** | one cell-face width 3.3e-05 too large at both walls. Override arm (substitute NEMO's own array, same entry state): collapse **16.2% basin / 24.9% wall rows / 9.1% south / 25.6% north** against registered bars of 50% owner / 10% refuted — PARTIAL on every registered reduction. Staggering control passes 292–307×. Mechanism confirmed (+0.750 against the arm's own response) but the shape test is NOT SUPPORTED against the measurement (−0.200). Band decomposition: 74% of the effect is in rows ≥185 with collapse **falling toward the wall** (83.5% at row 185 → 25.6% at the wall) — the candidate explains the wall row *worst*. **THE CONSTRUCTION DIFF IS NOW NAMED AND CLOSED.** It was never a different formula, radius or rounding: both models compute `R·Δλ·cos φ`, but NEMO evaluates φ at the V-point's own Mercator latitude (the transform taken at the HALF-INTEGER row index) while legoESM averaged the two adjacent tracer LATITUDES. On a Mercator coordinate `asin(tanh(·))` is nonlinear, so those are different latitudes — by up to 0.0011° here. legoESM already carried the correct face latitudes and was simply not using them for this one width. Fixed inside the existing NEMO-reproduction convention (no new switch; other grids' cards untouched). The corrected width reproduces NEMO's own array **bit-exactly** and is **BIT-IDENTICAL to the array the override arm substituted — 0 of 10,296 substituted cells differ on CPU; on GPU the same probe agrees to 1.1 ulp of float64 (2.5e-16 relative), which is 1.3e+11× smaller than the defect and is device `cos()` rounding, not a construction difference**, so the arm's five-state table above IS this fix's evidence **inside the barotropic substep loop the arm ran**, and no re-run is owed for that. **Scoped after review:** the builder fix also moves baroclinic-side consumers the arm never perturbed (lateral viscosity, GM/Redi, bolus, MLE, baroclinic EEN); all move toward NEMO and one is now exact (the F-point viscosity coefficient goes 3.3485e-05 → **0.0**), but their climate magnitude is unmeasured. The verdict is unchanged at PARTIAL: the fix removes a real infidelity, it does not promote the candidate. **It also EXPOSED a cancelling pair** — legoESM's vertex area is the spherical cap while NEMO divides by `e1f·e2f`, and with `e1f` corrected that pre-existing gap widens from 7.79e-06 to 2.22e-05 median. Fifth instance of the campaign's Rule-8 pattern — **and the FIRST of the five to be REFUTED as a pair by measurement (2026-08-27, §6c). CLOSED the same day.** The diff was a different QUADRATURE, not a different latitude: NEMO's `e1f·e2f` is the MIDPOINT value of `cos²φ` over the row while legoESM's cap is its EXACT interval integral, giving `(Δλ²/12)(3sin²φ − 1)` — reproduced at a measured/predicted ratio of **0.999984**, sign change and all. Fixed inside the same NEMO-reproduction convention, reusing the corrected v-face width (NEMO's `pphif == pphiv`, so its F-cell area IS that width squared); the corrected area matches NEMO's closed form to **3.1e-15**, from 4.10e-05. **Pair analysis, pre-registered at a bar of 1/10 and run BEFORE shipping:** the area and Coriolis halves meet only in the F-point absolute vorticity, where the area half is **1.51e-04** of the Coriolis half at the median (`|ζ|` sits four orders below `|f|`); in the lateral-viscosity channel `f` does not appear at all. **UNPAIRED-SAFE, shipped alone.** Cost, named: the discrete curl of solid-body rotation was exactly `f` on the old pair and is now +1.17e-05 off — still better than NEMO's own −2.73e-05. One-state fixed-bias response **INERT** (−0.011% wall, −0.040% deposit, against a pre-registered 1% bar), both toward NEMO. | `dino_wall_fixed_bias.md` §6a and §6b; issue #1455 "Override arm" |
| **Salinity advection** | EXONERATED | the "×10 worse at day 180" was a ratio-of-summed-magnitudes capped by sign-cancellation luck. Scored as rms-difference over rms-reference and split into horizontal and vertical parts (immune to their mutual cancellation), **every part improves at day 180** (full tendency 9.79e-3 → 8.87e-3; correlation 0.99995 → 0.99997). | issue #1455 "Operator debts #2/#3" |
| **Isoneutral slopes** | EXONERATED | feeding NEMO's own density into the production routine collapses all four slope rows **99.3–99.9%** at both states; the operator inherits its residual from a density input differing by 90–270 ulp. Slope error ~1e-13 against NEMO's own 1e-2 cap — cannot produce tenths of a Sverdrup. | issue #1455 "Operator debts #2/#3" |
| **Implicit vertical momentum solve** | FIXED, then EXONERATED for an exact reason | the geometry defect was real (half-thickness control volumes on 1560 u + 417 v closed faces; 1559 of 10,098 wet u-columns' depth divisor inflated up to 15.7%) and is fixed. Pre-registered 90-day A/B moves the ACC gap **0.00011 Sv** against a 0.129 bar. The reason is exact, not statistical: the barotropic split subtracts the column mean and adds it back, so the solve is divisor-independent there. | issue #1455 "Operator debt #1"; PR #1642 |
| **Barotropic time filter** | EXONERATED | per-step two-level rate and endpoint-to-endpoint circulation drift agree to **0.02%**, against a 23% deficit | `dino_basin_budget_result.md`, Decider 2 |
| **Face-thickness convention** | REFUTED as owner | accounts for about **1%** and is *anti-aligned* with the deficit; logged as convention debt | `dino_basin_budget_result.md` |
| **Rigid meridional position error** | REFUTED by its own residual | explains **0.8%** of the per-row shape difference | `dino_basin_budget_result.md`, Decider 0 |
| **Half-step impulse-response lag** | RETRACTED as a model finding | owned by the twin's **forward-Euler start** — predicted 0.500, measured 0.4702, falling to 0.000142 once bridged. (Mechanism wording as later corrected: a half-step delay **plus** a growing injected perturbation, ~0.002 Sv; the exact running-mean identity holds only for coincident restart time levels.) And the scare that followed was itself closed: **18 of 18 recorded gate/verdict runs already used the bridged start**; only ad-hoc CLI runs were Euler. Bounding A/B: start-mode difference 0.0181 Sv at day 90, inside the measured config-change envelope (0.0101–0.0704) and 5× under the gate floor. **Nothing re-baselines.** | issue #1455 "Centroid discriminator FINAL", "Euler-start scare CLOSED"; PR #1654 |

**The summary line the campaign wrote for itself**, after the post-tendency
bisection: *forcing exact · water masses exact · drag refuted (over-determined)
· lateral friction exonerated end-to-end · vorticity scheme fixed-and-inert ·
e3f inert · nonlinear tendency terms matched (with headroom) · divisor
fixed-and-inert · every post-tendency stage wrong-sign/wrong-shape at the
matched step · no net convergence (pure recirculation).*

**One amendment to that line, now itself superseded.** It was written before the
layer-versus-thickness weighting retraction, and this document then read
"lateral friction exonerated end-to-end" as PROVISIONAL pending a rescore.
**That caveat is RETRACTED (2026-08-27): the rescore had already landed three
days earlier and every friction leg survives it** — see the row above for the
numbers and for the two things the caveat got wrong. The line stands as
written. Every clause in it survives on evidence independent of the retracted
instrument.

**And the conclusion drawn from it:** the deficit is **not producible by any
operator difference at a matched state**. It must be **rectification** —
sub-floor per-step differences amplified through the basin's own feedback over
the year. That is the same tier the retention lane's "sustained quasi-equilibrium
difference (a feedback)" verdict predicted independently.

### 2.4 The surviving unknowns

Three, stated as open questions rather than leads.

1. **The banded fixed residual's owner.** What stands at the end walls is a
   genuine fixed field: step-to-step stable within 0.06–0.10% at the wall rows,
   and **banded, not wall-localized** — three alternating zonally-coherent
   basin-wide bands, peaking mid-basin, with the south wall indistinguishable
   from the interior (1.004×) and the north 12×. Geometry cannot explain it
   (identical latitude and spacing; the 14.7% depth difference predicts 1.15× of
   an observed 11.7×). Not grid-scale noise (zigzag content 0.3–9.4%).
   Localization by substitution came back **null**: bottom drag refuted, two
   damping terms inactive, slow forcing removes 3–4×, and **face depths /
   Coriolis / continuity / blend / filter all remain OPEN**. The one candidate
   with a mechanism — the v-face zonal metric, one cell-face width 3.3e-05 too
   large at both walls — is PARTIAL at ~25% and explains the wall row worst.
   **That width is now built the way NEMO builds it (2026-08-27), and the
   corrected construction reproduces the override arm's array bit-for-bit, so
   this remains PARTIAL — the fix removes the infidelity without moving the
   verdict, and the residual is still unowned.** Its
   supporting claim that the prediction "lands at **1.05–1.28×** the measured
   deficit with no fitted parameter" was itself **retracted in place**: it was
   two compensating errors, and the measured transfer gain is **0.66–0.77**, not
   1. The row-wise shape correlation is **−0.045** before substitution and
   **−0.200** after, against the measurement — NOT SUPPORTED — while against the
   arm's own response it is **+0.750**. Mechanism real; residual owned
   elsewhere.

   **Standing caveat, restated because it is easy to lose:** this fixed field
   **cannot** be the two-step flicker that opened the thread — a two-step
   projection annihilates fixed fields exactly. It is simply the largest
   remaining term at the wall rows.

2. **The end-wall sustained excess.** The ±69.5° zonal-wall excess survives
   bridging and sharpens. Un-bridged, the free-run 2Δt flicker stands at **17.95×
   the oracle**; bridged, at **2.89×** — an amplitude fall of about 6.2×
   (7.00e-6 → 1.13e-6 m). ("Drops 18×" would conflate the starting ratio with the
   size of the drop.) The bridged residual still concentrates **85% on the walls
   against NEMO's 4%**. Hypothesis 1 was RELOCATED, not refuted: a source-class
   difference on the end walls, about 2.5× smaller than first published. **j=1
   is the basin deficit's own wall row.** No suspect is named. Live candidates
   recorded: the barotropic split-explicit machinery, and the explicit-versus-
   implicit wind-stress placement at a wind-driven wall row.

3. **The fourth wind premise.** The wind-stress placement term survives
   exclusion only on a **four-premise structural argument, one premise of which
   is unverified**. It has been mislabelled three separate times — most recently
   as an unmeasurable dead term, when in fact its dump slot is allocated and
   never written, so the placement difference is real and measurable offline.
   Excluded at **neither** wall by size; the mid-task northern exclusion was a
   co-location artifact and is retracted.

Also open, and cheap: the **seasonal structure** of the basin deficit gets **no
verdict** — the best forcing-phase correlation is 0.48 against a registered
0.7/0.3 pair, and one cycle cannot separate "seasonal" from "not a straight
line". Season and elapsed time are degenerate in a single year.

---

## 3. Next-actions register, ranked with costs

### #1 — The Coriolis PAIR. Fix both halves together, never one.

> **RETRACTED 2026-08-26 — THIS ITEM'S CENTRAL PREMISE IS FALSE IN THE CHANNEL
> IT REGISTERS.** The claim below that the two errors "enter the same EEN
> rotation coefficient `e1v · f`" and "partially cancel" is true in the
> **zonal** tendency and false in the **meridional** one. Measured on the
> operator: perturbing `e1v` by +1 % moves the zonal tendency by 1.431e-06 and
> the meridional tendency by **exactly 0.000e+00** — the metric-complete EEN
> operator carries `e1v` into the zonal output and `e2u` into the meridional
> one. The wall-normal (meridional) residual this campaign scores therefore
> sees an **unpaired** Coriolis error, and there is nothing for the metric half
> to cancel against *in that coefficient*.
>
> **Scope it exactly:** "exactly zero" is true of the Coriolis operator and
> FALSE of the loop. `e1v` still reaches the meridional velocity through the
> continuity divergence and the ssh-average face depth, and the metric arm's
> meridional response is 58 % of the baseline residual's amplitude. What is
> refuted is the *cancellation mechanism*, not the metric's relevance.
>
> The three arms were run anyway and confirm it independently: the joint arm is
> the **sum** of its halves (`joint − sum` ≤ +6.2 points, ≤ +2.9 on every
> reduction but one), where a cancelling pair predicts it to exceed the sum.
> The two halves own **disjoint bands** — the metric owns the northern lobe
> (61.7 % against the Coriolis half's −1.5 %), the Coriolis owns both interior
> bands (32.1 % / 25.8 % against the metric half's −3.7 % / −2.8 %).
>
> Registered verdict on the joint arm: **PARTIAL**. Measured twice — first on
> the two-Earth tree (basin 32.3 %, lobe 61.8 %) and again after the
> rotation-rate fix, on the one-Earth model, where arm F is a clean convention
> arm (**basin 29.8 %, lobe 62.0 %**, OWNER bar 50 % on both). The staggering
> control fires at +12 808 %. The 90-day gate was **not** run, per the
> registered condition.
>
> **The Ω fix's own contribution is now separated and small**: it moves the
> baseline residual 3.62 % basin-wide, and it does so *entirely in the two
> interior bands* (−8.6 %, −7.0 %) while leaving the walls and the northern
> lobe untouched. Rate and convention compose multiplicatively to six digits
> (3.62 % and 6.55 % giving the old combined 9.93 %). So the campaign's
> hand-off residual on the corrected model is **1.7806e-07**, and the
> rotation-rate defect was never the bulk of it.
>
> **A separate finding in this item stands and was material:** the twin's
> geometry really was built on legoESM's rounded rotation rate while the config
> carried the card's pin — but the split is **config versus geometry**, not
> "card versus global constant", and the card's own pin cited NEMO's `key_cice`
> branch when DINO takes the sidereal one. Both are fixed. One rotation rate now
> reaches every site, bit-identical to the rate recovered from NEMO's own `ff_f`.
>
> Full account, with the operator measurement and the arm tables:
> **`dino_coriolis_pair_result.md`**. Everything below is the superseded
> framing, kept because the numbers in it are still the measured gaps.

**This is the ranked-first successor and it is not started.** *(Superseded: it
was run. See the retraction above.)*

legoESM builds the Coriolis parameter at the vertex as the **average of the two
adjacent tracer-row values**; NEMO evaluates it at **its own f-point latitude**.
Measured against NEMO's own dumped `ff_f` (`dino_wall_fixed_bias.md`, "The
sibling error this arm uncovered, and did not remove"):

| | median | RMS | at the walls | at the equator |
|---|---|---|---|---|
| v-face zonal metric gap (the arm just run) | **+1.86e-05** | 2.08e-05 | +3.35e-05 | +2.9e-09 |
| **Coriolis at the vertex** | **−5.48e-05** | **6.15e-05** | −2.50e-05 | −9.19e-05 |

The two are **exactly complementary**: the metric error peaks at the walls and
vanishes at the equator, the Coriolis error does the reverse, and their signed
latitude profiles correlate at **+1.000** (their magnitudes at −1.000 — the same
fact twice). In the coefficient, which goes as `e1v · f`, a positive relative
error in one and a negative one in the other **partially cancel**. The override
arm removed the **smaller** of the two.

That is precisely the Rule-8 pattern — raising faithfulness on one half of a
cancelling pair makes the metric worse — and this campaign recorded it four
times: the vertical-ladder paradox, seven faithful transcription fixes (the ACC
gap magnitude went 1.557 → 1.708 Sv), the reconciliation kernel (+0.048 Sv), and
the convection trigger (+0.032 Sv). **A confound rides on those magnitudes, and
it is this document's own rule applied to itself:** all three arm numbers were
recorded on the **old seasonal clock**, and §4's season rule says every arm
ranking recorded on that clock was superseded. The count of four is sourced; the
magnitudes have not been re-measured on the corrected clock. **Supersession, stated because it matters to how
much weight the pattern carries:** the first of those four, the ladder paradox,
was later overturned twice — first retracted as an artifact of the
surface-placement defect, then reversed outright when the twin was put on NEMO's
own ladders and the ACC gap *fell* from 1.87 to 0.60 Sv (PR #1638). Three
instances stand, not four. The pattern is real; it is not universal, and the
right response to it is a joint arm, never a revert. **Do not run the Coriolis
half alone.** The registered design is three arms — f alone, `e1v` alone (done),
and both together — with the verdict pre-registered on the **joint** arm.

**And a third of the Coriolis gap is not a discretisation convention at all.**
It decomposes into a uniform **−1.578e-05** and a latitude-varying **−3.90e-05**
median. The uniform part is a different Earth: legoESM's `constants.Omega` is
**7.292e-05**; NEMO's is **7.2921150830e-05** (2π over the sidereal day,
confirmed from NEMO's own `phycst.F90` and independently from the dumped
`ff_f`). Verified premise: legoESM's tracer latitudes equal NEMO's `gphit` to
1.4e-14 degrees, so this is a constant, not a grid difference. **An
oracle-matching card is running on a rotation rate the oracle does not use.**

**Code context, verified by reading the source during this consolidation** (not
a measurement, and not a claim about which sites are live in the run): the DINO
card already pins NEMO's rotation rate as a top-level `omega` field
(`experiments/dino.py:1299`, sourced from `NEMO_CONSTANTS_CONFIG.Omega =
7.292116e-05` in `ocean/constants_config.py:70`), and the campaign already fixed
one consumer of the 4-significant-figure literal — the eddy-coefficient path,
where Ω is squared into κ_GM (status board, fixed bug #14). Meanwhile
`packages/core/legoesm/constants.py:12` still reads `Omega = 7.292e-5`, and the
lat-lon C-grid Coriolis builder reads `getattr(grid, "omega", constants.Omega)`
(`ocean/dynamics/latlon_cgrid_operators.py:3278,3296`). So **the run plausibly
carries two rotation rates at once**, and the first move is to establish which
f-consuming sites see the card's pin and which fall back to the global constant.
That is a grep, not a run. Then decide whether the fix is a `ConstantsConfig`
pin, a card wiring change, or both.

**Cost:** the same freeze-and-vary offline substitution as the completed metric
arm, three arms. No new instrument. The premise check above is free.
**Verdict rule:** pre-register on the joint arm; the pair is the object.

### #2–#8 — Items already costed in the debt register

These carry over unchanged from `dino_outstanding_fidelity_debt.md` §D2 and from
the decisions the campaign escalated. Ranked by value-per-cost.

| # | action | why it exists | cost |
|---|---|---|---|
| **2** | **Split the gate's ACC row into channel + basin, scored separately.** | The full-section reducer sums a walled sub-polar gyre into a number labelled "circumpolar", and it is **opposite in sign to the channel it nominally measures** — the substep lane's own target had the wrong sign relative to the channel. The verdict run reached the same conclusion independently. This is a *proposal to the owner*: gate constants and reductions are not changed by a lane. | one gate-definition change + a re-score of recorded runs; no new compute |
| **3** | **Phase sweep + reference-antiphase control for the seasonal-clock attribution** (register D2.2/D2.3). | The clock A/B sampled only two phases (0 and 180 days), which cannot yield a share of the error; every percentage derived from it is retracted. Sound attribution needs a dose-response curve at 0/45/90/135/180 days and a control where the **reference** model runs antiphased forcing — zero model difference. Including a 90-day offset also closes the semiannual blind spot for free (an exact 180/360-day antiphase flips only odd harmonics). | 5 twin arms + 1 NEMO arm at 30 days each |
| **4** | **Re-calibrate bottom drag against the corrected drag** (register D2.1). | Two independent fixes weakened seafloor drag — the face-control-volume fix (`4734c2d5f`) and the factor-2 removal (#1645) — for a combined **~2× weaker in open water, ~4× weaker over stepped bathymetry**. No tuned value was changed in the merge, so every calibration inherited from before those commits is stale. | one coefficient sweep on the shipped card + a gate re-run |
| **5** | **Close the step-1 reconciliation gap for real** (register D2.4). | The forward-Euler first step returns before NEMO's second reconciliation site, so step 1 commits a depth-mean deposit NEMO removes. It now warns loudly instead of being silent, but the gap remains. Closing it is a **return-contract change**: `_step_impl` must surface the barotropic depth mean on its `_apply_implicit_vmix=True` path. | ~8 call sites + their tests |
| **6** | **fp64 snapshots on two ensemble members (days 30/90).** | fp32 snapshot storage is a live confound in two places: it bounds the measured ensemble spread from above, and it made "NEMO convects more often" a **storage artifact** (casting NEMO to fp32 reproduces 98.7–99.9% of the apparent gap; legoESM's true firing rate was never saved). Days 10–70 are currently unmeasurable at fp32 — 2 differing cells out of 342,134 at day 10. | rerun two existing members with fp64 snapshots; no model change |
| **7** | **The multi-year horizon.** | Everything certified is one year from a common state. The deep ocean has not adjusted, density metrics are all `no`, and two metrics (channel e3t_1d median, north of band) are recorded as **upper bounds only** because their ensemble spread is still growing fast enough that another year could close them. **REGISTERED, NOT RUN (2026-08-30):** `PREREG_multi_year_climate_equivalence.md` freezes the 20-year, six-member-per-side statistical-equivalence battery and horizon-matched floors; `dino_multi_year_climate_equivalence_handoff.md` holds the exact NEMO/legoesm arms. | run the registered held battery; the largest single compute item on the board |
| **8** | **The from-rest raise decision.** | Recorded as needing an owner's call, not a lane's. Kept here so it is not lost. | decision, then whatever it implies |

**Also on the board, no new work assigned** (each is a real observation with a
named cheap next step, none is ranked above the pair):

- ~~**The wall-row RESCORE**~~ — **DONE, and it was already done when this item
  was written.** It landed 2026-08-23; this board listed it as outstanding on
  2026-08-26 by quoting a to-do list in the source file rather than the section
  underneath that had executed it. Re-run at HEAD 2026-08-27 to confirm:
  **nothing reopens.** Friction's exoneration is RESTORED in §2.3, with the one
  leg that genuinely moved (enrichment 2.35× → 5.16×) reported there.
- The **grown-noise census** on the perturbation at days 5/10 in both models —
  states already on disk. The convective switch's ~300× rectification is
  established at ≳1e-10 perturbations and **silent at 1e-14**, so on current
  evidence the edge does not explain the models' amplification asymmetry at the
  ensemble kick amplitude; the census is the named conversion of that step to a
  measurement.
- The **wind-placement term**, now unlocked as measurable offline (its dump slot
  is allocated and never written).
- The **closed sub-basin mass budget** — the control volume nobody drew.
- The **78–435× kick-amplification asymmetry** (computational-mode probe named).
- The **T/S divergence atlas** finding: the difference field's amplitude
  flattens by day 90 but its **pattern keeps turning over** (84–114% field
  turnover from day 90 to 360), and in the southern basin it **migrates into the
  wall rows** (occupancy 0.033 → 0.741 against a 0.286 volume null). T
  difference 1.27e-2 K by day 360 (19× the chaos floor), S 7.9e-4 (13×).

**The feedback-tier chain, PLAUSIBLE end to end, each link labelled** — this is
the campaign's own best current story and it is explicitly not a verdict:
legoESM carries wall-enriched 2Δt flicker NEMO lacks (CONFIRMED) → noise grows
past ~1e-10 within days (measured growth curves) → convective adjustment's hard
edge rectifies it ~300×/step (CONFIRMED at the sampled state, but **shared with
NEMO**, so not differential) → tracer differences migrate into the wall rows
over the year (CONFIRMED) → the wall's westward return lobe **overshoots by 34%**
while the rest of the basin is right to 3% (the −0.95 Sv net deficit, CONFIRMED
as a characterization). Note the
faithfulness subtlety recorded with it: NEMO's own EVD is also a hard switch, so
the difference is **not the edge but the noise feeding it** — the faithful-fix
target is the 2Δt mode's source, with edge-smoothing usable only as a mechanism
diagnostic, never as a fix.

---

## 4. The instrument canon

Every rule below was earned by a measured failure in this campaign, and each one
cost real time before it was written down. They are listed with the failure that
produced them, because the failure is what makes the rule stick.

**Season clock.** The twin ran **half a year out of phase** with NEMO. The
corrected-clock baseline went from 0/5 to 4/5 gate metrics and density gaps
collapsed 77–99%. *Every arm ranking recorded on the old clock was superseded.*
Five twin probes carried the confound and were fixed; two southern-torque budget
probes' cross-model rows were voided pending re-run. (PR #1634.)

**fp32 is a policy, not a detail.** At fp32 the k33 heat-budget increment peaked
at exactly **1.00 ULP** with 95% of cells quantizing to zero, and was
*bit-identical* between two configurations whose underlying fields differ by
2e-3 — the reading was pure quantization. Separately, fp32 snapshot storage
manufactured a "NEMO convects more often" finding that reproduces 98.7–99.9%
just by casting NEMO to fp32. Conservation and cancellation-prone diagnostics
are gated at fp64, enforced by an AST test so no future driver can skip it.

**Provenance and dirty trees.** A shrink-only baseline seeded from a **mid-PR
working tree** certified nothing and stayed stale while *"289 commits changed
nothing"* (PR #1603). Every arm now
carries a producer SHA; a baseline written by a `+dirty` tree was re-run at the
clean tree and had to reproduce exactly before being cited.

**Euler start.** An Euler-started leapfrog trajectory is (unfiltered) the
two-point running mean of the true one — an identity to 1e-15 — so it *is* a
half-step delay at every frequency, and it manufactured a "half-step model lag"
finding that survived four candidate refutations before being caught. The
mechanism wording was then corrected again in-commit: it is a half-step delay
**plus a growing injected perturbation** (~0.002 Sv); the exact running-mean
identity holds only for coincident restart time levels.

**Injection ≠ accumulation.** A matched-state per-step injection divided by an
accumulated gap is a **category error**. Converting one to the other requires
the trajectory's **retention** of a one-step injection, which had never been
measured. The per-step deposit is positive at every one of ten states while the
gap's increments are negative in 8 of 9 windows (sign agreement 1/9, correlation
negative); integrated naively it gives +13.1 Sv against a needed −0.807. **All
prior "×budget" ratios were retracted as void.**

**Tendency ≠ transport.** Fixing the wall's *worst tendency error* — closing
96–98% of it, reproducing NEMO to 15 significant figures — moved the transport
by **1/15 of a floor**. The pre-registered falsification fired. The wall's
tendency error and the wall's transport error are different objects. Recorded in
its own source as the deepest instrument lesson since injection≠accumulation.

**The bar belongs to the statistic.** A floor measured on a **saturated 10-year**
ensemble applied to an **unsaturated 90-day** comparison was wrong by up to
15,500×. "At the floor" verdicts had to be re-scored wholesale, and three
retractions were themselves retracted because they had been scored with the same
unvalidated cross-horizon transfer they had flagged.

**Reduction cancellation.** A reduction can annihilate the signal it is aimed
at. The flicker lane's own budget null cancelled the hunted signal by up to
**80,000×**. A ratio-of-summed-magnitudes on a sign-changing field is capped by
cancellation luck — split it into parts that are immune to their mutual
cancellation before scoring. Rows weighted equally gave 43.0%/48.5% ownership
where physics-weighted rows gave 61.9%; five rows holding 0.6% of the spin-up
carried 72% of the deficit.

**Layer-versus-thickness weighting.** Every wall-row number in an entire hunt —
headline enrichments included — was **layer-averaged**, over-weighting the 10 m
surface layer about **54×** against the 545 m bottom. Properly thickness-
weighted, the wall error is **12.4× smaller and the sign inverts on three of the
four wall rows**, and two refutations had been decided on that instrument. Separately, "the deficit
grows with depth / 82% below 500 m" was the **layer thickness** growing from 10
to 450 m — an error that would have sent the next work item to vertical mixing.

**Leapfrog parity sampling.** A 10-day restart grid samples a **single leapfrog
parity** (about a 1% band). Documented now for every future time-axis design.
Leapfrog aliasing was excluded on five consecutive steps with a positive control
showing ~100× mobility on the same functional.

**Heterogeneous-band masking.** A dry-row sea level of exactly 0.0 sitting next
to a wet −0.97 m **inverted a verdict** (94% → 37%) before the stencil was
masked. An enrichment existed only because the equator row sat in the
denominator. The census that counted land in its denominator kept its
zero-count conclusion but lost its cross-model use.

**A fixed field is invisible to a 2Δt projection.** 95% of the end-wall residual
is a fixed field, and every two-step instrument cancels it **by construction** —
so a null from that family says nothing about it. Recorded with the explicit
corollary that the finer substep split is *not* the move, because it would
subdivide a 95%-invisible quantity.

**A code comment is a pointer, never a citable fact.** A probe gated itself
against a day-90 gap taken from a code comment. That number belonged to a
different run. Replaced with the value the verdict run itself printed.

**An identity is not a measurement.** A collapse test was promised to "exonerate
every other term in one number". It holds — because on this configuration it is
an **algebraic identity**, so everything arrives through it by construction. It
narrowed nothing, and the paragraph proposing it is recorded as *"wrong in the
way that wastes work."*

**A length scale is not a law.** The excess sea level against the wall decays
over about **112 km**, and the Munk frictional width computed from the oracle's
own viscosity is **87 km** — the coincidence that promoted lateral friction to
prime suspect. Both the law and the reading were then retracted: the Munk scale
`(A/β)^(1/3)` is derived for a **meridional** wall, and DINO's southern wall is
zonal, so β never enters the balance in a zonally closed basin. *"The 87 km it
produced is arithmetic."* And with a ~9 km deformation radius against 40 km
cells, any wall structure is resolution-limited to two or three cells anyway.

**Cannot-fail controls.** Caught repeatedly and by both reviewers: a telescoping
closure check that could not fail; a gate whose test surface was built from the
array it divides by; a one-variable gate that was three tautologies; a control
that perturbed a cell whose value is exactly zero; two self-comparing "controls";
a parity band that was 83% one point; a probe printing four hardcoded stale
numbers as fresh measurements. Every control must be shown to fire on a planted
violation.

**A double mask hides a broken control.** A land control claimed the band mask
was its only defence; the 3-D wet mask inside the volume weights had already
zeroed the planted poison before the band mask was consulted, so the control
passed for a reason unrelated to its claim. The **same defect class** had
already been found and fixed once in a sibling probe on the same branch.

**A probe double-count is not a model property.** A closure "failure" of 49 per
row was published as proof that the two models' term sets are not in bijection,
and it commissioned a work item to build a matched term correspondence. It was
an order-1e4 double-count of the oracle's barotropic operand inside the probe.
Both reviewers found it independently; the work item it generated was withdrawn.

**Guards can alibi each other.** Two guards were found each relying on the other
to catch the case neither caught. Resolved by deleting both and replacing them.

**A retraction that lives only in a commit message is not a retraction.** Caught
three times: the commit said the claim was withdrawn while the tool kept
printing it. Structurally fixed in the probes.

**One gate printing another gate's number under its own label** — both read
0.000e+00, which is exactly how that hides.

### The self-policing gates now shipped

The canon is not advice; most of it is enforced. Each of these **refuses** rather
than warns.

| gate | what it refuses |
|---|---|
| **Season stamp** | the gate refuses antiphase candidates outright — stampless or legacy-clock artifacts exit loudly. `DINO_GATE_ALLOW_LEGACY_CLOCK=1` is the historical-reproduction escape, and the result is then labelled *not a NEMO comparison*. |
| **Precision gate** | `require_fp64` on the box-budget accumulators, plus an AST test (`test_box_budget_drivers_fp64_gated`) that fails if any future driver constructs one without it. |
| **Provenance stamp** | producer SHA + inputs + effective flags on every artifact; run directories, NEMO SHA and binary sha256 recorded for the oracle side. |
| **Start-mode stamp** | the bridged start is the default everywhere, stamped from the **observed state** rather than the requested flag; Euler sits behind a loud banner and the gate withholds verdicts on explicit-Euler artifacts. |
| **Ladder-content hash** | stamps which vertical ladder a run used. Known limitation, recorded rather than oversold: it hashes the four 1-D reference arrays only, so two runs on different bathymetry hash identically (register D2.7). |
| **Blown-up-member refusal** | a blown-up ensemble member is now fatal instead of being silently discarded; member reuse checks the version. |
| **Wide-halo filter refusal** | a time-filter mode that silently returned sea level identically zero is now refused with a test. |
| **Dispatch hardening** | unknown scheme selections raise rather than falling through to a default. |

---

## 5. The ledger

### 5.1 Fixes shipped

Twelve defects found and corrected in the recent arc, each with a test and a
measured before/after. Eight are **NEMO-faithfulness** fixes (1, 2, 4–9); the
rest are a solver defect (3), a correctness consolidation (10), a harness
default (11) and an instrument defect (12). The honest headline is that **fixes 4–7, 9 and 10 are climate-inert** — each
moved its gate metric by less than its own floor and shipped on correctness, not
because it moved a number. Fixes 1, 2 and 8 plainly did move metrics, and are why
the card passes its gate at all.

| # | fix | measured effect | where |
|---|---|---|---|
| 1 | **The season clock** — the twin ran half a year out of phase with NEMO | gate 0/5 → **4/5**; density gaps collapse 77–99% | PR #1634 |
| 2 | **The twin stands on NEMO's own vertical ladders** (thickness and T-depth) | ACC gap 1.87 → 0.60 Sv; channel 2.93 → 0.29 Sv. The two halves fix different things — true thicknesses fix the barotropic transport, true T-depths the baroclinic — so neither alone suffices | PR #1638 |
| 3 | **The default free-surface solver could not take a step** — a weight dispatch unpacked three values from a four-value return | 17 → **1** failing tests across three barotropic suites; 16 recovered, none broken | PR #1639 / #1641 |
| 4 | **The implicit momentum solve halved cell thicknesses at closed faces** (1560 u + 417 v faces; 1559/10,098 wet u-columns' divisor inflated up to 15.7%). Review found two worse defects in the same solve: a ~5e9 matrix entry and NaN gradients in 36 of 60 — a live AD hazard | ACC 0.00011 Sv (bar 0.129) — correct, and inert for an exact reason | PR #1642 |
| 5 | **Barotropic bottom-drag time level** — legoESM built the rate from the velocity *after* the 3-D momentum update; NEMO builds it from the now level and freezes it across the substep window | in-loop constant +2.0198e-3 → +1.2622e-3 Sv/step against NEMO's +1.2606e-3 — **residual 0.1%, so the time level was the whole drag defect** | PR #1645 |
| 6 | **The implicit vertical-mixing matrix rebuilt the same drag coefficient from the after state** it was handed — a 0.76 m/s time-level error, 7× larger than #5 | 1.11e-01 → **0.0** m/s; deliberate null held, no leak into the barotropic loop | PR #1645 |
| 7 | **Implicit bottom-drag double application** — the timestep prefactor already carries NEMO's half-times-sum | halved; also removes a real cross-module inconsistency (the vertical solve ran 2× the two barotropic sites sharing the coefficient) | PR #1645 |
| 8 | **The NEMO-faithful barotropic reconciliation pair** shipped as the `kamm_mlf` card default | **5/5** on the 90-day gate at 0.411 Sv; 61.9% closure on the physics-weighted reduction | `1d5b19a1d` (PR #1640, subsumed) |
| 9 | **The EEN vorticity width-weighting** — NEMO width-weights the north-south transport and divides by the local width | closes 96–98% of the wall tendency error; the verbatim transcription reproduces NEMO's dumped tendency to 2.0e-20 m/s² (7.7e-15 of the field's RMS); **transport-inert**, shipped on faithfulness | PR #1648 |
| 10 | **The face-depth divisor**, one rule now shared repo-wide | climate-inert to 0.0000 Sv on the wall band; shipped on correctness | PR #1648 |
| 11 | **The bridged start is the default**, stamped from the observed state | start-mode difference 0.0181 Sv at day 90, 5× under the gate floor | PR #1654 |
| 12 | **The per-term diagnostics wrapper silently dropped the before-level state** for lateral friction, so every oracle per-term budget in the campaign compared now-versus-before for that term. The wrapper is `LatLonCGridOceanModel.tendencies_with_diagnostics`; the fix adds an `ldf_state` parameter that forwards NEMO's before-level state. Nineteen probes under the DINO validation directory call it and **eighteen still do not pass one** | registered prediction was a fall of at least 50×; measured **262×** (1.5947e-9 → 6.0875e-12 m/s²) | `513339cba`, branch `fidelity/dino-wall-flicker` |

Earlier in the campaign, and upstream of all of the above, the single largest
fix: **the surface forcing applicator wrote its increment to the now level
outside `model.step`, where the leapfrog combine cancels it** — retention
`(1−2γ)/(2(1−γ))` = 4/9, so **56% of every applied surface flux was discarded**.
With NEMO's placement, in a 20-year from-rest A/B differing in that one field:
ACC year 5 67.06 → **91.41** (NEMO 91.13), year 10 83.03 → **121.58** (NEMO
121.07), and the missing densest classes appear at 0.94–1.03× NEMO's volume.
**Carried with it, because its own review attached them:** the headline magnitude
was never independently reproduced, arm B overshoots 1.02–1.05× after year 10,
and the recorded review verdict is **RESULT QUALIFIED** — not clean. A
per-operator bar could never have found it — `tra_sbc` was AT BAR throughout,
and remains so, because the increment it computes was always correct. Only where
the integrator consumed it was wrong.

### 5.2 Retractions

Retractions are **large by design** here, and the campaign treats that as its
main quality signal rather than an embarrassment. **No campaign total is quoted**
— a summed figure is exactly the kind of derived number this campaign has
learned not to cite. What follows is the per-lane tally as each lane's own record
states it. Add them up yourself if you want an order of magnitude; do not put the
sum in a document.

| lane | retractions explicitly numbered in its own record |
|---|---|
| step walk (PR #1638) | 5 |
| southern basin (PR #1640) | 8 |
| floor recalibration | 3 (a retraction *of* three retractions) |
| full-section budget | 3 |
| barotropic substeps, time axis | 5 |
| wall LDF alignment | 5 |
| the EEN funnel | 6 (4 claims false + 2 probe defects) |
| T/S divergence atlas | 10 |
| kick asymmetry | 3 |
| flicker hunt | 11 |
| re-projection | 3 |
| fixed-bias characterization | 7 (5 the lane's own) |
| override arm | 2 |

Separately, **13** are catalogued in the debt register's own retraction table
(`dino_outstanding_fidelity_debt.md` §F). Further retractions are recorded
without a count — including the largest single one, the **dropped-wind harness
artifact**: three lanes built conclusions on replay probes that called
`model.step(surface_forcing=None)`, dropping the momentum wind the card routes
through that argument. Passing the wind collapsed the residual **646×**
(1.69e-8 → 2.6e-11, the fp64 floor). The retraction covered a "4.2e-3 m/step
deterministic eta injection", a "0.88 m²/s wall-concentrated transport bias",
and the entire causal chain built from them.

Also retracted at campaign scale, and worth knowing before re-investigating
anything: the **"25 Sv ACC deficit"** framing (the NEMO reference was still
climbing at +2.07 Sv/yr at year 20); the **vertical-ladder paradox** ("the truer
geometry tracks NEMO worse") — an artifact of the surface-placement defect; the
**"231% eta-invariant violation"** (two probe defects; the invariant holds to
2.3e-14 m); **"flat in time"** as the rate-deficit shape; and the **half-step
impulse lag** as a model finding.

The campaign's own scorekeeping, recorded mid-arc: *the instrument-defect ledger
stands at ~14; the hypothesis ledger at 0-for-9; the read/measure ledger keeps
winning.* And elsewhere: *8-for-8 on Fortran-reading, 0-for-7 on
residual-reasoning hypotheses — the next move is a read, not another
substitution arm.* That is the single most transferable strategic lesson in the
campaign.

### 5.3 Pull requests

All ten campaign pull requests are **merged**. (Three of them — #1648, #1650 and
#1654 — were still open when this consolidation was scoped; they merged on
2026-08-24, and the later state is the one recorded here.)

| PR | branch | state | subject |
|---|---|---|---|
| #1603 | `fidelity/dino-carryover` | MERGED 08-19 | option-threading ratchet was red at merge + 4 stranded carryover edits |
| #1629 | `fidelity/dino-carryover-2` | MERGED 08-20 | southern-deficit budget chain — the 6 commits stranded by #1603's early squash |
| #1634 | `fidelity/dino-carryover-3` | MERGED 08-21 | **the season bug** — corrected-clock baseline passes 4/5 (was 0/5) |
| #1638 | `fidelity/dino-step-walk` | MERGED 08-22 | the twin stands on NEMO's own vertical grid; + three self-policing harness gates |
| #1639 | `hotfix/default-barotropic-filter-unpack` | MERGED 08-22 | the default free-surface solver cannot take a step |
| #1642 | `fix/zdf-face-control-volume-main` | MERGED 08-22 | the implicit momentum solve halved cell thicknesses at closed faces |
| #1645 | `fix/baro-drag-time-level` | MERGED 08-23 | **the verdict** — the channel is statistically indistinguishable at one year |
| #1648 | `fidelity/dino-basin-budget-1yr` | MERGED 08-24 | the southern-basin hunt — the operator level exhausted by measurement |
| #1650 | `fix/pierre-review-triage` | MERGED 08-24 | disposition of the dual-review findings on #1640/#1634/#1645 |
| #1654 | `fix/twin-euler-start-default` | MERGED 08-24 | the bridged start is the default; the Euler-start scare is bounded |

Two parent branches were **closed as subsumed**, not abandoned: #1640
(`fidelity/dino-southern-basin`) and #1644. Nothing was lost — the ladder work a
squash-merge had silently reverted was verified present in `main`.

**Unmerged and outstanding: `fidelity/dino-wall-flicker`.** It sits directly on
`origin/main`, fully contains `fix/twin-euler-start-default`, and is the stack the
accompanying PR body covers. Also unmerged, carrying probe work:
`fidelity/dino-carryover`, `fidelity/dino-carryover-2`, `fidelity/dino-eta-waves`,
`fidelity/dino-ts-atlas`, `fidelity/dino-kick-asymmetry`,
`fix/zdf-face-control-volume`.

*(Commit counts are deliberately omitted. They move every time anything lands,
and a stale count in a hand-off is worse than no count —
`git rev-list --count origin/main..<branch>` is authoritative and takes a
second.)*

### 5.4 Review statistics

Every lane ran **two independent adversarial reviews** (code and physics), and
the record is that the reviews found real defects in essentially every round —
they were not ceremony.

| lane | rounds and outcome |
|---|---|
| barotropic substeps | rounds 1–4 **DO-NOT-SHIP** with 5 / 2 / 9 / 1 blocking findings; round 5 SHIP from both |
| re-projection | **five** adversarial reviews across three rounds, **all NO-SHIP**, all acted on; tests 13 → 44, and fourteen formerly-green mutations now fail |
| fixed-bias characterization | **four** rounds to SHIP; tests 21 → 69, every four-round mutation failing individually; two guards found alibiing each other, resolved by deletion; a post-SHIP delta of two further defects disclosed for review |
| wall LDF alignment | 1 critical + 6 major + **3 demanded retractions**, all fixed |
| salinity advection / slopes | 13 code-review findings, all fixed; physics review concurs, no fix |
| eta waves | code review no-ship ×2 → fixed; physics ship ×2; both reviewers reproduced every headline from raw data; 157 tests |
| flicker hunt | dual-reviewed through two rounds each; 51 tests |
| override arm | two independent reviews pre-citation, **both found and fixed real defects** — a one-variable gate that was three tautologies, and an unread arm stamp |
| southern basin | reviews do-not-ship → clean; the flip proven bit-identical to the measured 5/5 arm on a 1-day twin |

The recurring defect class, caught three separate times and now structurally
fixed in the probes: **a retraction living in the commit message while the tool
kept printing the claim.**

---

## 6. If you are picking this up tomorrow

1. Read §1.4 first. Knowing what the result does *not* claim prevents the most
   expensive category of follow-on error.
2. Read §2.3 before proposing any candidate. The exoneration table is the
   campaign's highest-value artifact and re-testing a row in it is pure cost.
3. Start at §3 item #1, and **run the pair, not a half**. Before spending any
   compute, do the free premise check named there: establish by grep which
   Coriolis-consuming sites read the card's pinned rotation rate and which fall
   back to the global constant.
4. Treat §4 as binding, not advisory. Every rule in it was bought with a
   retraction.
