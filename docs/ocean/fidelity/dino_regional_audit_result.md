# The equatorial and northern regional audit of the DINO verdict year

**Instrument:** `scripts/validate/ocean_fidelity/dino_1226/regional_audit.py`
**Pre-registration:** `scripts/validate/ocean_fidelity/dino_1226/PREREG_regional_audit.md` (`f83dc681e`, committed before any regional statistic existed)
**Artifact:** `/tmp/dino_regional_audit_final/regional_audit.json`, stamped `a583f3bf0` on a clean tree, with the full run log beside it
**Upstream results this refines, and does not re-derive:** `dino_verdict360_result.md` (`a1387f1f7`), `dino_ts_divergence_atlas.md` (`c754652a4`), `dino_campaign_synthesis.md`

Everything here is offline from recorded states. No model was stepped. CPU only.

**This document was rewritten after two independent adversarial reviews, both of
which returned DO-NOT-SHIP.** The code review found four critical defects in the
instrument, two proved by planted violation. The physics review **inverted the
headline**. Section 10 lists every retraction; read it before citing anything
above it.

---

## 1. The result, in one sentence

> From a shared NEMO day-180 restart, the largest single-row transport
> disagreement anywhere in the DINO section at one year is **on the equator, at
> 206 times its own measured run-to-run floor — the largest IN FLOOR MULTIPLES,
> not in Sverdrups** — and the reduction the campaign has been using reports that
> same band as indistinguishable, because the tropical current system cancels
> within it.

**The largest gap in Sverdrups is a different row**: row 1, against the southern
wall, at **−0.2773 Sv** — four times the equator row's magnitude, but only 23×
its own (much larger) floor. The two rankings answer different questions and
this document keeps them apart.

**And the same reduction underlies the campaign's best-known fidelity result.**
The circumpolar channel — matched to NEMO at 17 parts per million over a full
year — has **17 of its 35 rows over 5× their own floor** and retains **2 %** of
its own summed per-row magnitude. Whether that agreement is physical or
arithmetic is **open**, and settling it is the first thing §11 asks for.

This is a statement about **one year from a common ocean state**. It is not a
statement that the two models share a climate, and the deep ocean has not
adjusted on this timescale. Every floor is measured on **n = 4** members per side
and is a factor-of-two estimate.

## 2. What was opened, and why

The verdict run split the DINO section into three latitude groups and found the
shipped card right in the channel and wrong in the southern basin. Its third
group — "north of band" — is **150 of 199 rows**, from 44.6 °S to 69.9 °N: the
entire tropics and the whole northern hemisphere, reported as one number
(−0.0385 Sv). It had never been opened.

**The registered six-band partition** (absolute T-rows, so no later re-reading of
a latitude threshold can move them). P1 and P2 are `acc_driver_decomp.LAT_GROUPS[0]`
and `[1]` **taken by reference**, never re-typed; P3+P4+P5+P6 is exactly its
`[2]`, so this audit is a refinement of the recorded split and reproduces it by
summation.

| band | T-rows | latitudes |
|---|---|---|
| P1 south of band | 0–13 | south wall … −64.44° |
| P2 channel band (re-entrant) | 14–48 | −64.44 … −45.35° |
| P3 southern subtropics | 49–78 | −44.65 … −20.55° |
| P4 equatorial ±20° | 79–119 | −19.61 … +19.61° |
| P5 northern subtropics | 120–150 | +20.55 … +45.35° |
| P6 northern subpolar | 151–198 | +46.05 … +69.85° |
| E10 equatorial ±10° *(nested in P4, reported alongside, never summed with it)* | 89–109 | −9.95 … +9.95° |

**What these numbers are, said once.** Only the channel band is zonally
re-entrant, so only its number is a **throughflow**. Every other band is a
zonally closed basin, where this reduction is a longitude-averaged section flux —
a volume-mean zonal velocity times an area — and nothing is conserved through it.
The **gap** between two models under the identical functional is the meaningful
quantity; the absolute value is not a current.

## 3. THE HEADLINE FINDING — the band reduction was hiding the equator

### 3.1 The measurement

Scored **per row** rather than per band, with each row's floor measured from the
same two 4-member ensembles, at day 360:

| row | latitude | gap [Sv] | that row's floor [Sv] | ×floor |
|---|---|---:|---:|---:|
| **99 (the equator)** | 0.00° | **−0.0704** | 0.000342 | **206** |
| 98 | −0.70° | −0.0335 | 0.000232 | 145 |
| 104 | +3.5° | +0.0143 | 0.000139 | 103 |
| 100 | +0.70° | −0.0326 | 0.000330 | 99 |
| 197 (north wall) | +69.50° | −0.0233 | 0.000288 | 81 |
| 96 | −2.1° | +0.0324 | 0.000650 | 50 |

**22 of the 41 rows in the equatorial band exceed 5× their own floor; 36 of 41
exceed the 2× INDISTINGUISHABLE bar.** The band's own registered escalation rule
fires on more than half of it.

### 3.2 Why the band number said the opposite

| band, day 360 | net [Sv] | Σ\|per-row gap\| [Sv] | net / Σ\|row\| |
|---|---:|---:|---:|
| P1 south of band | −0.9519 | 1.3138 | **−0.725** |
| P2 channel band | +0.0024 | 0.1229 | **+0.020** |
| P3 southern subtropics | −0.0397 | 0.0537 | **−0.740** |
| **P4 equatorial ±20** | **+0.0059** | **0.4020** | **+0.015** |
| P5 northern subtropics | −0.0081 | 0.0331 | −0.246 |
| P6 northern subpolar | +0.0034 | 0.1022 | +0.034 |
| E10 equatorial ±10 | +0.0055 | 0.3706 | +0.015 |

The equatorial band's net retains **under 3 %** of its own summed per-row
magnitude. The sum of the per-row gaps reproduces the band number to **5e−15 Sv**,
so this is not two instruments disagreeing — it is one instrument cancelling.

**Quote the magnitude, not the sign.** Across the four member pairs that ratio
runs **+0.015 / +0.022 / −0.013 / −0.015** at day 360: its spread exceeds its
own value, so the *sign* of the residue is undetermined and only its
*magnitude* is measured. That is all the argument needs — a band retaining a
couple of per cent of its own summed per-row magnitude is not measuring how far
apart the two models are, whichever way the residue points. (At day 90 the sign
is stable.)

**Every band that reads INDISTINGUISHABLE in the table below sits at 1.5–3.4 %.
Both bands that read as gaps sit at 72–74 %.** The three "clean" bands are clean
in the reduction, not in the ocean.

And nothing about the equatorial disagreement decays: Σ\|row gap\| across the band
is **0.425 / 0.447 / 0.384 / 0.402 Sv** at days 90/180/270/360, and the equator
row's own gap is **−0.0845 / −0.0854 / −0.0774 / −0.0704**. Only the cancellation
improves.

This is the campaign's own **"reduction cancellation"** rule — *a reduction can
annihilate the signal it is aimed at* — firing on this audit. It was caught by
the physics review, not by me, and the per-row ledger and the cancellation column
are now committed so it cannot recur silently.

## 4. The band verdict table — correct, and now readable only with its caveat

Gap = legoESM − NEMO, control member, **mean** reduction over longitudes. Floor
is `sqrt(spread_lego² + spread_NEMO²)`, each a sample std over that side's **own**
four members **in that band at that day**. **No global floor is transferred.**
Registered rule: **INDISTINGUISHABLE ⟺ |gap| ≤ 2 × floor.**

| band | gap [Sv] | floor [Sv] | ×floor | verdict | rows >5× | net/Σ\|row\| |
|---|---:|---:|---:|:--|---:|---:|
| P1 south of band | −0.9519 | 0.0617 | 15.42 | **gap-at-15.4×** | 11/14 | −0.725 |
| P2 channel band | +0.0024 | 0.0096 | 0.25 | INDISTINGUISHABLE | **17/35** | **+0.020** |
| P3 southern subtropics | −0.0397 | 0.0069 | 5.80 | **gap-at-5.8×** | 14/30 | −0.740 |
| P4 equatorial ±20 | +0.0059 | 0.0061 | 0.98 | INDISTINGUISHABLE | **22/41** | **+0.015** |
| P5 northern subtropics | −0.0081 | 0.0025 | 3.22 | gap-at-3.22× | 19/31 | −0.246 |
| P6 northern subpolar | +0.0034 | 0.0039 | 0.89 | INDISTINGUISHABLE | **41/48** | **+0.034** |
| E10 equatorial ±10 | +0.0055 | 0.0040 | 1.36 | INDISTINGUISHABLE | **16/21** | **+0.015** |

"rows >5×" counts that band's rows exceeding **5× their own** floor.

**The four INDISTINGUISHABLE verdicts are statements about the functional.** They
are correct as scored and they must never be quoted without the last column.

**Every band is UNSATURATED at day 360** — the ensembles are still growing. As
registered, an INDISTINGUISHABLE still stands and a `gap-at-N×` is an upper
bound. The materiality test (imported from `verdict360`, not re-invented) says
only P1's `no` survives that fact unaided.

**Whose floor is it?** legoESM's own spread is 79 % (P3), 87 % (P4) and 98.5 %
(P1) of the two-sided floor's variance in the transport metric. In the *T/S*
metric the equatorial upper class **inverts** this — see §6.

The **day-360 escalation gate** fires on **P1 (15.42×) and P3 (5.80×)**. Applied
across all four horizons the same rule escalates all seven bands, because the
day-90 floors are three to five orders under their day-360 values; that form is a
cannot-fail gate and is reported as such rather than kept for form.

## 5. Finding 2 — "north of band" is neither tropical nor northern, but it is not simply P3 either

The verdict run's single number decomposes at day 360 as **P3 −0.0397, P4
+0.0059, P5 −0.0081, P6 +0.0034**, summing to **−0.0385 Sv** — reproducing the
recorded value, with the six bands summing to **−0.98797 Sv** against the
recorded full-section −0.9880.

**P3 alone is 103 % of it**; the other three cancel to +0.0012 Sv. Reporting 150
rows as one number attributed a 30-row signal to the tropics and the northern
hemisphere.

**That +0.0012 Sv is itself a cancellation, and must not be read as agreement.**
The equatorial +0.0059 Sv inside it is 1.5 % of 0.402 Sv of per-row signal on the
control member, and under 3 % on every member (§3.2). The three bands cancel *as reduced numbers*; their rows do not.

### 5.1 The channel's 17 parts per million is also a cancellation

**This is the most consequential measurement in the audit and it is not about
P3.** The circumpolar channel band — the campaign's headline fidelity result,
`+0.0015 Sv on a 34 Sv transport, 17 ppm`, reproduced here unchanged — has
**17 of its 35 rows exceeding 5× their own floor** and a net that retains
**+0.020** of its own summed per-row magnitude.

The verdict run's channel number is **not disputed**: it is the same functional
on both sides and the gap is valid. What is new is that it survived a reduction
which discards ~98 % of the per-row magnitude in that band, exactly as the
equatorial band's does. **Whether the channel's agreement is physical or
arithmetic is open**, and this audit does not settle it — it only shows that the
question exists and that the standing evidence cannot answer it.

### 5.2 Two more bands this audit measured and does not resolve

Both are reported rather than deferred silently, because each is the same
cancellation signature as the headline.

**P6, the northern subpolar band, has the highest fraction of escalating rows in
the section: 41 of 48 rows (85 %) exceed 5× their own floor**, including the two
rows against the north wall at **81×** and **72×**, while the band net is
+0.0034 Sv at 0.89× — INDISTINGUISHABLE — with net/Σ|row| = **+0.034**. That is
the equatorial signature in a different basin, and this audit does not explain
it.

**P5, the northern subtropics**, is the only other band with a resolvable band
gap (−0.0081 Sv at 3.22×), and it too is bottom-referenced-led. It is reported
for completeness and is not investigated here.

**But three of the words I first used for P3 do not survive scrutiny.**

**"Stable" is a property of the sum, not of the field.** The net is flat
(−0.038/−0.040/−0.035/−0.040) while Σ\|row\| inside P3 grows 33 %, and the
physics review's sub-block decomposition shows the three thirds of the band
moving by 3.7×, 1.4× and a sign reversal, summing to a constant.

**About half of it is a displaced front.** A single-parameter rigid meridional
shift fitted over rows 40–70 against NEMO's own per-row profile explains **55 %
of the variance at day 360** (correlation 0.74; 0.61/0.61/0.75/0.74 across the four
horizons; the fit is through the origin, so the figure quoted is the fraction of
the gap's own sum of squares removed, **not** `corr²`) for a displacement of **0.011 rows ≈ 0.9 km, about 1 % of a grid
cell**. The registered P2/P3 boundary sits close to the structure's zero
crossing, so where the band edge falls changes how the signal splits between
"clean channel" and "P3 gap".

**"Barotropic" is the wrong word.** The split is `bt = u_bottom·H`, which
`acc_driver_decomp`'s own docstring calls the *reference-level* transport and
"NOT a true depth mean". What the ordering table does support, against the
margin's own bar, is:

| band, day 360 | \|gap bt\| | \|gap bc\| | margin | 2× its own floor | verdict |
|---|---:|---:|---:|---:|:--|
| band | \|gap bt\| | \|gap bc\| | margin | 2× its own floor | flips | verdict |
|---|---:|---:|---:|---:|---:|:--|
| P1 south of band | 2.4782 | 1.5262 | −0.9519 | 0.1126 | 0 | *(no vertical verdict — margin IS the gap)* |
| P2 channel band | 0.0499 | 0.0523 | +0.0024 | 0.0182 | 2 | **UNRESOLVED** |
| **P3 southern subtropics** | 0.0373 | 0.0024 | −0.0349 | 0.0142 | 0 | **bottom-referenced-led (2.5× the bar)** |
| P4 equatorial ±20 | 0.0528 | 0.0587 | +0.0059 | 0.0153 | 2 | **UNRESOLVED** |
| P5 northern subtropics | 0.0231 | 0.0149 | −0.0081 | 0.0062 | 0 | *(no vertical verdict — margin IS the gap)* |
| P6 northern subpolar | 0.0299 | 0.0333 | +0.0034 | 0.0038 | 0 | **UNRESOLVED** |

**The tool now REFUSES a directional verdict on the IDENT rows** rather than
printing one and caveating it, because a printed verdict is what gets quoted:
on those rows "bottom-referenced-led" would restate "this band has a gap". Only
P3 gets a direction.

**This table is WEAK evidence, and the reason is algebraic.** `bt + bc == gap`
by construction, so wherever the two legs carry **opposite signs** — 6 of these
7 rows — `|bc| − |bt|` **is the band gap itself**. Those rows re-score §4's own
number against a different bar whose severity varies 23× across the table; they
are not independent evidence about vertical structure. Only P3, whose legs share
a sign, is a genuinely separate statement.

The bar above is the **margin's own** floor — the sample std of the four
member-pair margins. An earlier version RSS'd the two legs' floors, which is
wrong for a quantity that is not a difference of independent parts; both
reviewers found it independently. The corrected bars are 1.03× to 23× *smaller*
and no verdict flips, but P3's ordering weakens honestly from **5.2× its floor**
to **2.5× the 2× bar** — two different denominators, named here because quoting
them side by side without saying so is how a weakening reads as a collapse.
Where the margin *is* the gap the two floor constructions differ by at most 2×
with no systematic direction, so the correction is immaterial on those rows; it
matters only on P3, the one row that is a separate statement.
For the equatorial band, **2 of the 4 member pairs take the opposite sign** —
measured, not asserted.

**Registered-rule deviation, disclosed:** the pre-registration fixes every floor
as the RSS of the two *sides'* spreads. The margin does not factorise that way —
it is a single cross-model quantity, so there is one ensemble of it and its floor
is that ensemble's standard deviation. This is the only statistic in the audit
whose floor is not the registered RSS, and it deviates by disclosure.

**What is honestly left of the P3 finding:** the rows immediately north of the
channel carry a coherent (net/Σ\|row\| = −0.74) gap of −0.0397 Sv at 5.8× its own
floor, about half of which is consistent with a ~1 km displacement of the ACC's
northern flank. It is the only *coherent* band gap outside the southern basin
(net/Σ|row| = −0.74), and its bottom-referenced character clears its own bar at
2.5× the registered bar — the one row of the ordering table that is not the
band gap in disguise.

## 6. Finding 3 — the equatorial current structure

**A ~30 % shear error and a displaced wind-driven layer — not a 10 % weak jet.**

| horizon | shear 5→26 m, legoESM | NEMO | ratio | undercurrent top, legoESM | NEMO | deeper by |
|---|---:|---:|---:|---:|---:|---:|
| day 90 | +0.01548 | +0.02178 s⁻¹ | 0.711 | 34.9 m | 32.0 m | +3.0 m (+9.3 %) |
| day 180 | +0.01460 | +0.02147 | 0.680 | 34.5 m | 31.1 m | +3.4 m (+10.9 %) |
| day 270 | +0.01448 | +0.02029 | 0.714 | 33.1 m | 30.1 m | +3.0 m (+10.0 %) |
| day 360 | +0.01427 | +0.01983 | 0.719 | 31.6 m | 28.4 m | +3.3 m (+11.5 %) |

The profile difference is a **dipole**: +0.0445 m/s at 5 m and **−0.0720 m/s at
26 m**, where NEMO reads only −0.0308 — legoESM's near-surface westward flow
there is over three times NEMO's. Both lobes are at **166×** and **103×** the
ensemble floor of that same statistic. The top three levels (0–31.4 m) hold
**77 %** of the thickness-weighted column difference.

**The physical statement is a deeper, less sheared wind-driven layer**: the same
wind-driven momentum spread over a thicker, more slab-like surface layer, stable
to ±0.4 m across four horizons.

**No mechanism is claimed.** DINO's equatorial dynamics are known to be
vertical-mixing sensitive, and this signature is *consistent with* a near-surface
viscosity difference of tens of percent — but a **profile is not a term**, and
the equatorial upper ocean is also where the largest temperature divergence in
the domain sits, co-located with it. A mixing-depth error changes temperature and
a temperature error changes mixing; from state snapshots alone neither can be
separated. Labelled **PLAUSIBLE**. §11 names the measurement that would settle
it.

## 7. Finding 4 — the T/S divergence, re-reduced

Volume-weighted rms of (legoESM − NEMO), **thickness-weighted by `e3t_0`**, never
layer-averaged. Day 360, as multiples of each band-and-class's **own** floor:

`*` marks a floor whose two sides differ by over a decade; UNMEAS marks one at
or under 10× its own fp32 storage quantum, where the multiple would be measuring
the file format and is withheld (which is **not** a pass).

| band | upper <200 m | interior 200–1400 m | abyss >1400 m |
|---|---:|---:|---:|
| P1 south of band | 4.4× | 3.7× | 2.8× |
| P2 channel band | 40.8× | 2.7× | 7.4× |
| **P3 southern subtropics** | **214×** | **415×** | UNMEAS |
| P4 equatorial ±20 | 24.2×`*` | 8.9× | 12.6× |
| P5 northern subtropics | 7.1×`*` | 57.8×`*` | UNMEAS |
| P6 northern subpolar | 9.4×`*` | 6.8×`*` | 19.6×`*` |
| E10 equatorial ±10 | 24.1×`*` | 7.9× | 8.3× |

**This ranking is partly a ranking of denominators, and the table now says so.**
Share of each floor's variance that is legoESM's **own** dispersion:

| band | upper <200 m | interior | abyss |
|---|---:|---:|---:|
| P1 south of band | 88.3 % | 86.4 % | 70.3 % |
| P2 channel band | 74.9 % | 77.3 % | 94.4 % |
| P3 southern subtropics | 77.6 % | 87.3 % | 85.0 % |
| **P4 equatorial ±20** | **0.2 %** | **3.7 %** | 51.1 % |
| P5 northern subtropics | 100.0 % | 99.9 % | 98.1 % |
| P6 northern subpolar | 99.9 % | 99.8 % | 99.8 % |

The equatorial upper class is **the one place in the domain where NEMO is the
noisier model**: its floor there is **99.8 % NEMO's own dispersion**, which
*inverts* the verdict run's standing limitation #1 (that 77–99 % of the combined
floor is legoESM's spread). A large multiple in another band is partly telling
you where NEMO is quiet, not only where legoESM is wrong. P3's floor, by
contrast, is genuinely two-sided at 78–87 %.

**Does each band hold its own divergence pattern?** The registered statistic —
forward retention of the day-10 hotspot set against a 0.05 geometric null — **is
void in the band it was used on**, and it took **two** controls to see why.

A **depth-only** mask scores **0.9985** in the equatorial band against the day-10
set's **0.9794**. But that control is itself too weak: it ranks cells
monotonically from the surface, whereas the divergence peaks at the
**mixed-layer base**, so a hotspot set can beat it by locating the right *level*
— vertical information dressed as horizontal.

The honest control is a **horizontally uniform** field carrying the day-10
**per-level** volume-weighted rms, cut by the identical 5 %-of-volume rule. It
keeps the true vertical profile and removes every horizontal cue, so a set that
beats it is using horizontal information and nothing else. Ranked within the
upper class, at day 360:

| band | dT retention | LEVEL | ×LEVEL | RANDOM null | ×RANDOM | ceiling |
|---|---:|---:|---:|---:|---:|---:|
| **P4 equatorial ±20** | 0.643 | 0.123 | 5.2× | 0.103 | **6.3×** | 0.969 |
| E10 equatorial ±10 | 0.440 | 0.125 | 3.5× | 0.096 | 4.6× | 0.933 |
| **P5 northern subtropics** | 0.581 | 0.092 | **6.3×** | 0.148 | 3.9× | 0.761 |
| P3 southern subtropics | 0.091 | 0.036 | 2.5× | 0.033 | 2.7× | 0.570 |
| P2 channel band | 0.099 | 0.096 | 1.0× | 0.053 | 1.9× | 0.557 |
| P6 northern subpolar | 0.073 | 0.068 | 1.1× | 0.051 | 1.4× | 0.373 |
| **P1 south of band** | 0.049 | 0.044 | **1.1×** | 0.051 | **1.0×** | **0.830** |

**Two denominators, because the first one degenerates here.** Inside a single
depth class the LEVEL baseline is nearly the field's own ranking — the upper
class is three levels with over 90 % of its cells in one — so the shipped table
also carries a **measured** null: the mean retention over 200 random sets drawn
with the hot set's **own per-level volume**, which cannot be beaten by knowing
*which* level and scores horizontal placement alone. The published ratios are
**conservative** against it (P4 5.2× → 6.3×).

**The ceiling column kills the obvious objection.** The southern basin's
retention of 0.049 is not "its divergence went quiet": its day-360 field
retains **0.830** of its own day-360 hotspot set. The divergence there is
strongly concentrated — it has simply **moved**. That is the difference between
*no memory* and *nothing to remember*, and only the ceiling separates them.

**These ratios carry NO measured ensemble floor.** Both LEVEL and RANDOM are
**structural** baselines, not n = 4 floors. Ratios of 5× and above are safe
against any plausible ensemble floor; **the 1.0–2.5× ordering is UNSCORED**, and
no band inside that range may be ranked against another.

**Two things change from the first version of this document.**

First, three bands I credited with horizontal memory have little or none: P1,
P2 and P6 sit at **1.0–1.9×**, where I printed 12×, 3× and 0.9× against the
weaker baseline.

Second — and this corrects a factual error in the intervening version — **the
result is not equatorially specific, and the two tracers do not agree on the
leader.** Temperature puts the northern subtropics ahead against the LEVEL
baseline (6.3× vs 5.2×); salinity puts the equator ahead and by a wider margin
(11.0× vs 4.7×); against the measured RANDOM null the equator leads in both. So:

> **The equatorial and northern-subtropical upper oceans both hold their
> horizontal divergence pattern, at 5–11× a horizontally-uniform baseline,
> against 1.0–1.2× for the southern basin. Which of the two leads depends on the
> tracer and on the baseline, so this audit does not rank them.**

No mechanism is attached to that. It is tempting to call it a wind-driven
surface-layer property, and this document previously did — but P3 and P5 are
mirrored bands under a symmetric wind and they score 2.7× and 3.9×, so a
symmetric forcing does not explain an asymmetric result. **The hemispheric
asymmetry is the interesting question this audit leaves open**; that the
southern flank is the one stirred by the ACC is a hypothesis, not a finding.

**All three baselines are kept in the shipped table**, because their contrast is
what demonstrates that stratification was doing the work in the registered
statistic.

## 8. The registered predictions, scored

| # | prediction | verdict | the number |
|---|---|:--|---|
| R1 | the equatorial band carries a transport gap over its own floor at day 360 | **REFUTED as scored, and the scoring is the finding** | band +0.0059 Sv at 0.98× — but per row the equator is at **206×** and 22 of 41 rows clear 5× |
| R2 | P1 is still the largest absolute band gap | CONFIRMED | P1 at 0.9519 Sv |
| R3 | the equatorial gap is predominantly baroclinic | **UNRESOLVED** | margin +0.0059 Sv against a 2× bar of **0.0153** (the margin's own floor); **2 of 4 member pairs flip the sign** — measured |
| R4 | the equatorial T divergence holds its own hotspots | **CONFIRMED, but restated and NOT equatorially specific** | as registered 0.9794 against a 0.9985 depth and 0.9979 level baseline (void); level-controlled 0.643 vs 0.123 = 5.2×, and 6.3× against a measured random null. **The northern subtropics scores as highly (6.3× on temperature), and which band leads depends on the tracer and the baseline**, so the two are not ranked |
| R5 | P6 gap is smaller than P5 | CONFIRMED | 0.0034 vs 0.0081 Sv |

## 9. Controls

All controls are green in the shipped run, and **six fired on real defects**
during development or review. K9, the known-answer control, reproduces the
verdict run's recorded day-360 south-of-band gap of **−0.95191 Sv to 2.2e−06 Sv**,
compared in code, and aborts on a planted wrong value.

| control | what happened |
|---|---|
| **K3b dry-row census** | found row 198 (north wall) fully dry, mirroring row 0. The two walls are a NAMED exception with a MEASURED reason — **K3c** poisons both entire wall rows and requires every statistic to move by exactly 0.0. |
| **K5 dry-cell plant** | fired at 2.132e−14 Sv, which was **not** a mask leak: the NEMO loader returns a `moveaxis` view and `.copy()` makes it C-contiguous, so `einsum` sums in a different order. Both arms now come from one contiguous baseline and the layout sensitivity is printed. |
| **jet empty-level refusal** | DINO's tropical bathymetry does not reach the deepest level; the profile is truncated at the deepest wet level and the truncation is printed. |
| **K3 day-0 identity** *(found by code review)* | was `\|x\| ≤ 10\|x\|` — the legoESM day-0 snapshot IS `fp32(NEMO day-0)`, so gap and quantum were the same number, printing a ratio of 1.000000 in all seven bands. Replaced by an exact bit-identity, which passes on 342134 T/S cells and 336338 u cells and **can** fail. |
| **K5b T/S plant** *(found by code review)* | was a **double mask** — the "dry" cell it planted sat at 4253 m, killed by the depth mask before the wet mask was consulted. Replaced by a plant against what this file owns: the band-and-depth intersection, poisoned from another band and another depth class. |
| **the registered band cuts** *(found by code review)* | a one-row shift of the equatorial band passed the **entire** self-test. The three registered cuts are now asserted at import against the f = 0 row; the reviewer's exact plant now fails before the module loads. |

**The structural zero:** `f` is **exactly 0.0** at T-row 99. This probe computes
no vorticity-projection, no Coriolis-normalised and no f-weighted statistic
anywhere, so the row is excluded by never entering a denominator; the
denominators it *does* enter are printed to show they are not structurally zero.

**Provenance:** the producer revision is captured at the start of the run and
re-checked before the artifact is written. A tree that moves mid-run is now
refused — the code review caught an earlier artifact stamped with a commit that
landed while the run was in flight.

## 10. Retractions

**Both adversarial reviews returned DO-NOT-SHIP on the first version of this
document. Every claim below was in it.**

| # | claim as first written | status |
|---|---|---|
| **1** | "the equatorial bands are INDISTINGUISHABLE; R1 is refuted; the tropics are clean" | **RETRACTED — inverted.** True of the band reduction, false of the ocean. Per row the equator is at 206× its own floor and 22 of 41 rows clear 5×; the band net retains under 3 % of its own summed per-row magnitude. |
| **2** | "legoESM's equatorial westward surface jet is ~10 % weak" | **RETRACTED — understated ~3×.** It is a ~30 % shear error with a 9–11 % deeper wind-driven layer. |
| 3 | "the top two levels are ~37 m of a 4506 m column" | **RETRACTED — arithmetic error.** They span 0–20.6 m; 37 m is level 3's centre depth. The top three hold 77 % of the thickness-weighted column difference. |
| 4 | "the cancellation is **why** the transport reads indistinguishable" | **DOWNGRADED to PLAUSIBLE.** The two functionals differ by the per-level wet count and the width weighting; consistency is not demonstration. |
| **5** | "the equatorial T divergence holds its own hotspots (0.979 vs a 0.05 null)" | **RETRACTED as registered.** A depth-only mask scores 0.9985 — the registered statistic was measuring stratification. Re-established only on a **level**-controlled statistic (0.643 vs 0.123 = 5.2×). |
| **5b** | the depth-controlled ranking "P4 26×, P2 12×, P3 10×, P6 3×" | **RETRACTED.** The depth-only baseline is monotone from the surface while the divergence peaks at the mixed-layer base, so a set could beat it with vertical information. Against the **level** baseline and a **measured random null**: P4 5.2×/6.3×, P5 6.3×/3.9×, E10 3.5×/4.6×, P3 2.5×/2.7×, and P1/P2/P6 at 1.0–1.9×. The effect is **not equatorially specific** and the two bands are **not ranked**. |
| **5f** | "salinity gives the same ordering more strongly" | **RETRACTED — factually wrong.** Salinity **inverts** the leader (P4 11.0× vs P5 4.7×, where temperature has P5 6.3× vs P4 5.2×), and P2/P6 double to 2.2× in salinity, so "no horizontal memory at all" was temperature-only. |
| **5g** | "a property of the wind-driven upper ocean" | **WITHDRAWN as a mechanism label.** P3 and P5 are mirrored bands under a symmetric wind and score 2.7× and 3.9×; a symmetric forcing does not explain an asymmetric result. The hemispheric asymmetry is left open. |
| **5c** | R3 scored against an RSS-of-legs bar | **RETRACTED.** `bt + bc == gap`, so with opposite-sign legs the margin **is** the band gap and the RSS bar was up to 23× too large. Re-scored on the margin's own floor; no verdict flips, but P3 weakens 5.2× → **2.5×**. |
| **5d** | "the band net retains 1.5 %" quoted with its sign | **QUALIFIED.** Across members the ratio is +0.015/+0.022/−0.013/−0.015: the **sign is undetermined** at day 360. Magnitude only. |
| **5e** | "two of four members flip it" (R3) | **WAS A CODE COMMENT, NOT A MEASUREMENT.** Now computed and printed; it happens to be correct. |
| 6 | "P3's gap is REMARKABLY STABLE" | **RETRACTED as physics.** The sum is stable; Σ\|row\| grows 33 % and the internal composition rotates. |
| 7 | "P3's gap is almost entirely BAROTROPIC" | **RELABELLED.** It is bottom-referenced, per the reducer's own docstring, and every band shares a negative bt gap; P3 is where the compensation is absent. The ordering does clear its bar. |
| 8 | R3 "CONFIRMED" | **RETRACTED to UNRESOLVED.** The margin is half the floor of either leg, and two of four members flip it. |
| 9 | "P3 carries the largest T/S divergence in floor-multiples" | **QUALIFIED.** True of the ratio; the equatorial denominator is 99.8 % NEMO's own dispersion. |
| 10 | the day-90 escalation list (all seven bands) | **WITHDRAWN as a gate.** A gate that fires on everything decides nothing; the day-360 endpoint is the gate. |
| 11 | the first F4 colour scale | **WITHDRAWN.** Set by one 0.472 K cell against a 0.068 K 99.5th percentile. |
| 12 | "reaches 234 % at depth" | **WITHDRAWN.** Divided by a zero-crossing profile. |

Also withdrawn, from this instrument's own development: a depth-only baseline
that first scored exactly 0.0000, because `hotspot_set` ranks by **absolute**
value and a `−gdept0` ranking field put the masked cells first.

**Nothing from an upstream document is retracted.** The verdict run's −0.0385 Sv
and −0.95191 Sv are reproduced exactly. What changes is what may be attributed to
them.

## 11. What the next lane should do first

1. **Re-score the channel's 17-parts-per-million result per row.** It is the
   campaign's best-known fidelity claim, and this audit finds the channel band
   sitting at net/Σ|row| = **+0.020** with **17 of its 35 rows over 5× their own
   floor**. Whether that agreement is physical or arithmetic is open, and it is
   the single highest-value hour in this list. The same applies to every
   standing INDISTINGUISHABLE verdict in the campaign. Offline, minutes.
2. **Explain P6.** The northern subpolar band has the highest escalating-row
   fraction in the section (41 of 48, 85 %), with the two north-wall rows at 81×
   and 72×, and a band net that reads INDISTINGUISHABLE. This audit measured it
   and does not explain it.
3. **The matched-state substitution for the equatorial layer.** Feed NEMO's own
   day-360 T, S, u, v at the equator into legoESM's shipped vertical-mixing
   closure and compare the resulting mixing coefficient against NEMO's dumped
   `avm_k`, level by level, over the top 40 m. If they differ at a matched
   state, the closure owns it; if they match, the difference is inherited from
   the thermal state. **This is what breaks the temperature-and-mixing
   circularity, and until it runs no sentence about this gap may contain the
   word "mixing".** The NEMO restarts carry `avm_k`, `en`, and the full momentum
   trend decomposition.
4. **A ten-minute first cut:** a fixed-density-threshold mixed-layer depth at the
   equator row, same criterion both sides, all 20 horizons.
5. **Do not tune the equatorial transport.** It is at 0.98× its own floor while
   the profile behind it is at 166×, and the band number retains under 3 % of
   that band's summed per-row magnitude. Optimising the number that is already
   right would move the structure in an unconstrained direction — the same trap
   the campaign recorded for lateral viscosity and channel transport.
