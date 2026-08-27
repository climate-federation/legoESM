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
> 206 times its own measured run-to-run floor** — and the reduction the campaign
> has been using reports that same band as indistinguishable, because the
> tropical current system cancels within it.

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

The equatorial band's net retains **1.5 %** of its own summed per-row magnitude.
The sum of the per-row gaps reproduces the band number to **5e−15 Sv**, so this
is not two instruments disagreeing — it is one instrument cancelling.

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

| band | gap [Sv] | floor [Sv] | ×floor | verdict | net/Σ\|row\| |
|---|---:|---:|---:|:--|---:|
| P1 south of band | −0.9519 | 0.0617 | 15.42 | **gap-at-15.4×** | −0.725 |
| P2 channel band | +0.0024 | 0.0096 | 0.25 | INDISTINGUISHABLE | **+0.020** |
| P3 southern subtropics | −0.0397 | 0.0069 | 5.80 | **gap-at-5.8×** | −0.740 |
| P4 equatorial ±20 | +0.0059 | 0.0061 | 0.98 | INDISTINGUISHABLE | **+0.015** |
| P5 northern subtropics | −0.0081 | 0.0025 | 3.22 | gap-at-3.22× | −0.246 |
| P6 northern subpolar | +0.0034 | 0.0039 | 0.89 | INDISTINGUISHABLE | **+0.034** |
| E10 equatorial ±10 | +0.0055 | 0.0040 | 1.36 | INDISTINGUISHABLE | **+0.015** |

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

**But three of the words I first used for P3 do not survive scrutiny.**

**"Stable" is a property of the sum, not of the field.** The net is flat
(−0.038/−0.040/−0.035/−0.040) while Σ\|row\| inside P3 grows 33 %, and the
physics review's sub-block decomposition shows the three thirds of the band
moving by 3.7×, 1.4× and a sign reversal, summing to a constant.

**About half of it is a displaced front.** A single-parameter rigid meridional
shift fitted over rows 40–70 against NEMO's own per-row profile explains **55 %
of the variance at day 360** (correlation 0.74; 0.61/0.61/0.75/0.74 across the
four horizons) for a displacement of **0.011 rows ≈ 0.9 km, about 1 % of a grid
cell**. The registered P2/P3 boundary sits close to the structure's zero
crossing, so where the band edge falls changes how the signal splits between
"clean channel" and "P3 gap".

**"Barotropic" is the wrong word.** The split is `bt = u_bottom·H`, which
`acc_driver_decomp`'s own docstring calls the *reference-level* transport and
"NOT a true depth mean". What the ordering table does support, against the
margin's own bar, is:

| band, day 360 | \|gap bt\| | \|gap bc\| | margin | 2× its own floor | verdict |
|---|---:|---:|---:|---:|:--|
| P1 south of band | 2.4782 | 1.5262 | −0.9519 | 0.7769 | bottom-referenced-led |
| P2 channel band | 0.0499 | 0.0523 | +0.0024 | 0.4465 | **UNRESOLVED** |
| P3 southern subtropics | 0.0373 | 0.0024 | −0.0349 | 0.0133 | bottom-referenced-led |
| P4 equatorial ±20 | 0.0528 | 0.0587 | +0.0059 | 0.0350 | **UNRESOLVED** |
| P5 northern subtropics | 0.0231 | 0.0149 | −0.0081 | 0.0051 | bottom-referenced-led |
| P6 northern subpolar | 0.0299 | 0.0333 | +0.0034 | 0.0078 | **UNRESOLVED** |

**What is honestly left of the P3 finding:** the rows immediately north of the
channel carry a coherent (net/Σ\|row\| = −0.74) gap of −0.0397 Sv at 5.8× its own
floor, about half of which is consistent with a ~1 km displacement of the ACC's
northern flank. It is the only *coherent* band gap outside the southern basin.

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
void in the band it was used on**. A **depth-only** mask, chosen with zero
knowledge of the pattern, scores **0.9985** in the equatorial band against the
day-10 set's **0.9794**: the set's horizontal information content there is
negative, and it was measuring stratification.

Ranked **within the upper class**, which removes the depth information and asks
the question the prediction meant to ask:

| band | dT retention, day 360 | its depth-only baseline | ratio |
|---|---:|---:|---:|
| **P4 equatorial ±20** | **0.643** | 0.025 | **26×** |
| E10 equatorial ±10 | 0.440 | 0.025 | 18× |
| P5 northern subtropics | 0.581 | 0.014 | 41× |
| P2 channel band | 0.099 | 0.008 | 12× |
| P3 southern subtropics | 0.091 | 0.009 | 10× |
| P6 northern subpolar | 0.073 | 0.023 | 3× |
| **P1 south of band** | **0.049** | 0.052 | **0.9× — at its baseline** |

**So the substance survives, on the controlled statistic only:** the equatorial
divergence holds its horizontal pattern (26× its baseline) while the southern
basin sits *at* its baseline, i.e. has no horizontal memory of where it was at
day 10 — reproducing the atlas's own finding for P1 by an independent route.

## 8. The registered predictions, scored

| # | prediction | verdict | the number |
|---|---|:--|---|
| R1 | the equatorial band carries a transport gap over its own floor at day 360 | **REFUTED as scored, and the scoring is the finding** | band +0.0059 Sv at 0.98× — but per row the equator is at **206×** and 22 of 41 rows clear 5× |
| R2 | P1 is still the largest absolute band gap | CONFIRMED | P1 at 0.9519 Sv |
| R3 | the equatorial gap is predominantly baroclinic | **UNRESOLVED** | margin +0.0059 Sv against a 2× bar of 0.0350; two of four members flip it |
| R4 | the equatorial T divergence holds its own hotspots | **CONFIRMED, on a statistic the registered one had to be replaced by** | as registered 0.9794 vs a 0.9985 depth baseline (void); depth-controlled 0.643 vs 0.025 |
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
| **1** | "the equatorial bands are INDISTINGUISHABLE; R1 is refuted; the tropics are clean" | **RETRACTED — inverted.** True of the band reduction, false of the ocean. Per row the equator is at 206× its own floor and 22 of 41 rows clear 5×; the band net retains 1.5 % of its own summed per-row magnitude. |
| **2** | "legoESM's equatorial westward surface jet is ~10 % weak" | **RETRACTED — understated ~3×.** It is a ~30 % shear error with a 9–11 % deeper wind-driven layer. |
| 3 | "the top two levels are ~37 m of a 4506 m column" | **RETRACTED — arithmetic error.** They span 0–20.6 m; 37 m is level 3's centre depth. The top three hold 77 % of the thickness-weighted column difference. |
| 4 | "the cancellation is **why** the transport reads indistinguishable" | **DOWNGRADED to PLAUSIBLE.** The two functionals differ by the per-level wet count and the width weighting; consistency is not demonstration. |
| **5** | "the equatorial T divergence holds its own hotspots (0.979 vs a 0.05 null)" | **RETRACTED as registered.** A depth-only mask scores 0.9985 — the registered statistic was measuring stratification. The claim is re-established on the depth-controlled statistic (0.643 vs 0.025) and only there. |
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

1. **Score the campaign's standing verdicts per row, not per band.** This audit's
   central lesson is that the band reduction cancels; the verdict run's own
   INDISTINGUISHABLE channel result deserves the same cancellation column
   (P2 sits at +0.020). Offline, minutes.
2. **The matched-state substitution for the equatorial layer.** Feed NEMO's own
   day-360 T, S, u, v at the equator into legoESM's shipped vertical-mixing
   closure and compare the resulting mixing coefficient against NEMO's dumped
   `avm_k`, level by level, over the top 40 m. If they differ at a matched
   state, the closure owns it; if they match, the difference is inherited from
   the thermal state. **This is what breaks the temperature-and-mixing
   circularity, and until it runs no sentence about this gap may contain the
   word "mixing".** The NEMO restarts carry `avm_k`, `en`, and the full momentum
   trend decomposition.
3. **A ten-minute first cut:** a fixed-density-threshold mixed-layer depth at the
   equator row, same criterion both sides, all 20 horizons.
4. **Do not tune the equatorial transport.** It is at 0.98× its floor while the
   profile behind it is at 166×, and the band number is 1.5 % of its own signal.
   Optimising it would move the structure in an unconstrained direction.
