# The equatorial and northern regional audit of the DINO verdict year

**Instrument:** `scripts/validate/ocean_fidelity/dino_1226/regional_audit.py`
**Pre-registration:** `scripts/validate/ocean_fidelity/dino_1226/PREREG_regional_audit.md` (`f83dc681e`, committed before any regional statistic existed)
**Artifact:** `/tmp/dino_regional_audit/regional_audit.json`, stamped `17881d91d` (clean tree), with the full run log beside it
**Upstream results this refines, and does not re-derive:** `dino_verdict360_result.md` (`a1387f1f7`), `dino_ts_divergence_atlas.md` (`c754652a4`), `dino_campaign_synthesis.md`

Everything here is offline from recorded states. No model was stepped. CPU only.

---

## 1. The result, in one sentence

> From a shared NEMO day-180 restart, one year of legoESM drift away from NEMO
> is **indistinguishable from the two models' own run-to-run spread in the
> tropics and in the northern subpolar ocean**, and the verdict run's single
> "north of band" number is **not tropical and not northern at all** — it is
> essentially the thirty rows immediately north of the circumpolar channel. But
> the *transport* statistic that says so is **the wrong functional for the
> equator**: the equatorial current profile differs by up to 234 % of its own
> ensemble floor in the top 50 m, and the depth integral cancels that away.

This is a statement about **one year from a common ocean state**. It is not a
statement that the two models share a climate, and the deep ocean has not
adjusted on this timescale. Every floor is measured on **n = 4** members per
side and is a factor-of-two estimate; a band at 2–3× its floor is not resolved.

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

Both ±10° and ±20° are reported because neither is the honest equatorial width
alone: DINO's own analytic wind carries an equatorial dip of Gaussian width
7.5°, ±10° is roughly the tropical cell and ±20° its subtropical flanks.
Reporting one width would be a choice presented as a fact.

## 3. THE VERDICT TABLE — band × horizon

Gap = legoESM − NEMO, control member, **mean** reduction over longitudes (the
median reduction is not additive over bands and is never used here). Floor is
`sqrt(spread_lego² + spread_NEMO²)`, each a sample std over that side's **own**
four members **in that band at that day**. **No global floor is transferred to
any band.**

| band | day 90 | day 180 | day 270 | **day 360** |
|---|---|---|---|---|
| P1 south of band | −0.426 (2937×) | −0.232 (10.4×) | −0.057 (0.92×) | **−0.9519 (15.4×)** |
| P2 channel band | +0.070 (1009×) | −0.005 (6.7×) | −0.028 (3.9×) | **+0.0024 (0.25×)** |
| P3 southern subtropics | −0.038 (1602×) | −0.040 (44.3×) | −0.035 (15.8×) | **−0.0397 (5.80×)** |
| P4 equatorial ±20 | +0.035 (44820×) | +0.022 (8.3×) | +0.009 (4.5×) | **+0.0059 (0.98×)** |
| P5 northern subtropics | −0.020 (4164×) | −0.013 (11.2×) | −0.007 (8.4×) | **−0.0081 (3.22×)** |
| P6 northern subpolar | +0.007 (3428×) | +0.004 (5.2×) | +0.001 (0.86×) | **+0.0034 (0.89×)** |
| E10 equatorial ±10 | +0.030 (69149×) | +0.025 (13.3×) | −0.003 (2.00×) | **+0.0055 (1.36×)** |

Gaps in Sv; the multiple is of **that band's own** two-sided floor. The
registered rule: **INDISTINGUISHABLE ⟺ |gap| ≤ 2 × floor.**

**Day-360 verdicts, spelled out:**

| band | gap [Sv] | floor [Sv] | ×floor | verdict |
|---|---:|---:|---:|:--|
| P1 south of band | −0.9519 | 0.0617 | 15.42 | **gap-at-15.4×** |
| P2 channel band | +0.0024 | 0.0096 | 0.25 | **INDISTINGUISHABLE** |
| P3 southern subtropics | −0.0397 | 0.0069 | 5.80 | **gap-at-5.8×** |
| P4 equatorial ±20 | +0.0059 | 0.0061 | 0.98 | **INDISTINGUISHABLE** |
| P5 northern subtropics | −0.0081 | 0.0025 | 3.22 | **gap-at-3.22×** |
| P6 northern subpolar | +0.0034 | 0.0039 | 0.89 | **INDISTINGUISHABLE** |
| E10 equatorial ±10 | +0.0055 | 0.0040 | 1.36 | **INDISTINGUISHABLE** |

**How to read the day-90 column.** Every multiple there is in the thousands
because the 1e-14 kick has barely grown: the day-90 floors are three to five
orders under their day-360 values. That is the campaign's own "the bar belongs
to the statistic — **and to the horizon**" rule, and it is why the day-360
endpoint is the scored one. It is **not** a finding about seven bands.

**Two flags travel with every row, and both are asymmetric on purpose.**
*`u`* — the band's ensemble is UNSATURATED (its spread is still growing over the
last two quarters, measured per side exactly as `verdict360.saturation_table`
does it). **Every band in this table is unsaturated at day 360.** As registered,
an INDISTINGUISHABLE still stands (the runs did not separate even with room to);
a `gap-at-N×` does not — N is an upper bound.
*`u!`* — unsaturated as a fact, but the growth it actually shows could not close
the gap in another year, so the flag does **not** void that row's verdict. This
is `verdict360.unsaturated_materiality`, imported rather than re-invented,
because "not saturated" otherwise voids every `no` in the table for free, which
is not honest.
At day 360 **P1 alone carries `u!`**: it needs its floor to grow 7.7× and it is
growing at 1.02× per quarter. Its `gap-at-15.4×` stands.

*`1s`* — the two sides' spreads differ by more than one decade, so the RSS floor
is numerically one model's own dispersion. At day 360 no band trips it, but the
underlying disclosure still applies: **legoESM's own spread is 79 % (P3), 87 %
(P4) and 98.5 % (P1) of the two-sided floor's variance.** The floor is mostly
one model's wobble, and that is disclosure, not a different formula.

## 4. Finding 1 — "north of band" is **not** tropical and **not** northern

The verdict run's single number decomposes, at day 360, as:

| band | gap [Sv] | share of the −0.0385 |
|---|---:|---:|
| **P3 southern subtropics** | **−0.0397** | **103.2 %** |
| P4 equatorial ±20 | +0.0059 | −15.3 % |
| P5 northern subtropics | −0.0081 | +21.0 % |
| P6 northern subpolar | +0.0034 | −8.9 % |
| **sum** | **−0.0385** | **100 %** |

and the six bands sum to **−0.98797 Sv**, reproducing the recorded full-section
−0.9880 Sv.

**P3 — the thirty rows immediately north of the channel — carries the whole of
it.** The other three bands together contribute **+0.0012 Sv**, i.e. they cancel
each other. Reporting 150 rows as one number attributed a 30-row signal to the
tropics and the northern hemisphere, where there is none to attribute.

**P3's gap has two properties the southern basin's does not.**

**It is stable.** −0.0384, −0.0396, −0.0348, −0.0397 Sv at days 90/180/270/360 —
a **3 % spread across the whole year**, where P1 swings −0.43 → −0.23 → −0.06 →
−0.95. P3 looks like a **fixed offset**; P1 looks like a trajectory.

**It is barotropic, and cleanly so.** Splitting each band into the
reference-level part (`bt = u_bottom·H`) and the shear (`bc = ∫(u−u_bottom)dz`),
which sum identically to the band total:

| band, day 360 | total | bt | bc |
|---|---:|---:|---:|
| P1 south of band | −0.9519 | **−2.4782** | **+1.5262** |
| P3 southern subtropics | −0.0397 | **−0.0373** | −0.0024 |
| P4 equatorial ±20 | +0.0059 | −0.0528 | +0.0587 |

P3's shear part is **6 % of its total and never exceeds 0.0024 Sv at any
horizon**; its barotropic part is essentially the whole gap and is as stable as
the total. P1, by contrast, is a **pair of ~2 Sv terms cancelling to −0.95** —
the same cancellation structure the campaign already recorded at day 270, here
seen at every horizon. P4's total is likewise a cancelling pair, and section 5
shows what that cancellation is hiding.

**Read `bt` as its own docstring says**: it is the *reference-level* transport,
what a depth-uniform field equal to the deepest wet value would carry. It is
**not** a depth mean and must not be read as one.

**Status: this is a NEW REGIONAL FINDING and it triggered the registered
escalation.** At the day-360 endpoint the bands over 5× their own floor are
exactly **P1 (15.42×) and P3 (5.80×)**.

**What it is NOT, stated because it is the obvious wrong reading:** P3 being
adjacent to the channel does not make it the channel's problem. P2's own gap at
day 360 is +0.0024 Sv at 0.25× its floor — the channel is the best-matched band
in the table. Whether P3 is an independent object or the southern system's flank
is **not settled by this audit** and is named as the open question in section 8.

## 5. Finding 2 — the equator: a structural gap the transport cannot see

### 5.1 The transport says indistinguishable, and it is right about itself

P4 is **+0.0059 Sv at 0.98× its own floor**, E10 **+0.0055 Sv at 1.36×**. The
registered prediction R1 — that the equatorial band would carry a resolvable
transport gap — is **REFUTED**.

### 5.2 The current structure says something quite different

Zonal-mean `u(z)` over wet u-cells on the equator row, day 360, against that
statistic's **own** measured ensemble floor:

| level | depth | legoESM | NEMO | difference | ×floor |
|---|---:|---:|---:|---:|---:|
| 0 | 5 m | −0.4013 | −0.4458 | **+0.0445** | **166** |
| 1 | 15 m | −0.2586 | −0.2458 | −0.0128 | 14 |
| 2 | **26 m** | **−0.1027** | **−0.0308** | **−0.0720** | **103** |
| 3 | 37 m | +0.0968 | +0.1090 | −0.0122 | 54 |
| 4 | 49 m | +0.1618 | +0.1586 | +0.0032 | 67 |
| 6 | 74 m | +0.1553 | +0.1549 | +0.0003 | 13 |

**The difference is a DIPOLE, not a weakening.** legoESM's westward surface jet
is 10.0 % weak at 5 m — and at 26 m its westward flow is **−0.1027 against
NEMO's −0.0308, more than three times NEMO's value**. The jet is **displaced
downward**: the sign change sits at 5–15 m in legoESM's difference field, and
the eastward undercurrent's upper flank is correspondingly late.

Both lobes are far outside the floor. The surface lobe is at **166×** and the
26 m lobe at **103×** the ensemble spread of that same statistic, and the
surface relative deficit is essentially **constant all year**: −9.8 %, −10.1 %,
−9.9 %, −10.0 % at days 90/180/270/360.

Below about 50 m the two profiles agree to under 1 % of the profile's own
surface magnitude, and below 74 m to 0.3 % in absolute terms.

### 5.3 Why both statements are true at once — the number that reconciles them

Integrating the difference over the top 200 m with the true layer thicknesses —
which is what a depth-integrated transport actually sees:

| band, day 360 | positive lobe | negative lobe | net | cancellation |
|---|---:|---:|---:|---:|
| equator row | +0.4936 | −1.1372 | −0.6436 | 57 % of the larger |
| E10 ±10° | +0.1116 | −0.1203 | **−0.0087** | **7.2 % of the larger** |

On the ±10° band the two lobes cancel to **1.3–17.4 % of the larger** across the
four horizons — i.e. **83–99 % of the signal is annihilated by the vertical
integral**. The top two levels are ~37 m of a 4506 m column and hold **31–38 %**
of the summed column difference.

**The transport metric is not wrong; it is the wrong functional for this
difference.** This is the campaign's "reduction cancellation" lesson and its
"tendency ≠ transport" lesson arriving together in a third form: **structure ≠
transport**. A band can be INDISTINGUISHABLE in Sverdrups and plainly different
in its currents, and only reporting both catches it.

### 5.4 The blind spots, and how the instrument was built around them

`f` is **exactly 0.0** at T-row 99 (asserted and printed by control K11;
neighbours ±2.545e−06 s⁻¹). The campaign has already withdrawn a published
3.51× enrichment that existed *only* because that row sat in a denominator where
the vorticity flux structurally vanishes.

**This probe computes no vorticity-projection, no Coriolis-normalised and no
f-weighted statistic anywhere.** Row 99 is therefore excluded from nothing *by
exclusion* — it is excluded by never entering a denominator. The denominators it
*does* enter are printed so a reader can see they are not structurally zero:
transport width 1.112e+05 m, summed wet thickness 1.880e+05 m, row wet volume
2.365e+15 m³, 1697 of 1872 wet u-cells.

## 6. Finding 3 — the T/S divergence, re-reduced

Volume-weighted rms of (legoESM − NEMO), **thickness-weighted by `e3t_0`**,
never layer-averaged. Day 360, as **multiples of each band-and-class's own
measured floor**:

| band | upper <200 m | interior 200–1400 m | abyss >1400 m |
|---|---:|---:|---:|
| P1 south of band | 4.4× | 3.7× | 2.8× |
| P2 channel band | 40.8× | 2.7× | 7.4× |
| **P3 southern subtropics** | **214×** | **415×** | **467×** |
| P4 equatorial ±20 | 24.2× | 8.9× | 12.6× |
| P5 northern subtropics | 7.1× | 57.8× | 82.2× |
| P6 northern subpolar | 9.4× | 6.8× | 19.6× |
| E10 equatorial ±10 | 24.1× | 7.9× | 8.3× |

**Amplitude and anomaly point at different bands, and the anomaly is the one
that matters.** In raw kelvin the equator dominates — 9.07e−02 K in P4's upper
class against P3's 1.44e−02 K, which is the atlas's "~98 % equatorial-upper"
result. Measured against each band's **own** chaotic floor, **P3 is an order of
magnitude more anomalous than the equator in every depth class**, and it is the
same band that carries the transport gap. The atlas's own lesson — that the
leading raw box is also the leading noise box — reproduces here per band, and it
points at P3.

**Does each band hold its own divergence pattern?** Forward retention of the
band's day-10 hotspot set (5 % of that band's own volume, ranked by intensity;
geometric null 0.05, registered bar 0.25):

| band | dT ret d90 | d180 | d270 | **d360** |
|---|---:|---:|---:|---:|
| P1 south of band | 0.376 | 0.133 | 0.073 | **0.054** |
| P2 channel band | 0.733 | 0.859 | 0.856 | **0.832** |
| P3 southern subtropics | 0.958 | 0.928 | 0.860 | **0.830** |
| **P4 equatorial ±20** | 0.988 | 0.987 | 0.981 | **0.979** |
| P5 northern subtropics | 0.960 | 0.895 | 0.852 | **0.928** |
| P6 northern subpolar | 0.836 | 0.846 | 0.701 | **0.738** |
| E10 equatorial ±10 | 0.983 | 0.982 | 0.978 | **0.979** |

**The equator holds its own divergence more strongly than any other band** —
0.979 against a 0.05 null, essentially unchanged all year — while the southern
basin collapses to **0.054, exactly the null**, reproducing the atlas's finding
that P1 abandons its day-10 hotspot from day 210 onward.

**So the tropical divergence is locally generated and stationary, not advected
in.** Registered prediction R4 is CONFIRMED. Salinity gives the same ordering
(P4 0.920, P1 0.054).

## 7. The registered predictions, scored

| # | prediction | verdict | the number |
|---|---|:--|---|
| R1 | the equatorial band carries a transport gap over its own floor at day 360 | **FAILED** | P4 +0.0059 Sv at **0.98×** — refuted |
| R2 | P1 is still the largest absolute band gap | CONFIRMED | P1 at 0.9519 Sv |
| R3 | the equatorial gap is predominantly baroclinic | CONFIRMED | P4 bt −0.0528 vs bc +0.0587 Sv |
| R4 | the equatorial T divergence holds its own hotspots | CONFIRMED | retention 0.979 vs bar 0.25, null 0.05 |
| R5 | P6 (subpolar) gap is smaller than P5 (subtropics) | CONFIRMED | 0.0034 vs 0.0081 Sv |

R1 was the audit's headline hypothesis and it is **refuted by its own registered
bar**. R3 is confirmed but is a knife-edge on two nearly-cancelling terms, and
section 5.3 is what that cancellation actually means — do not read R3 as "the
equatorial gap is a baroclinic transport error".

## 8. What this audit did NOT establish

* **No mechanism is claimed for anything here.** The equatorial dipole is
  *consistent with* a near-surface vertical-structure difference, and DINO's
  equatorial dynamics are known to be vertical-mixing sensitive — but this audit
  measured a **profile**, not a **term**, and a profile difference cannot name
  its own owner. Labelled **PLAUSIBLE**, not confirmed, and not to be repeated
  without the word.
* **Whether P3 is an independent object or the southern system's flank is
  open.** The discriminating measurement is a per-row walk of the gap across
  rows 40–90 with the channel/basin boundary marked: an object of the channel's
  should decay away from it, a fixed offset of P3's own should not.
* **The 10.0 % surface-jet deficit is not attributed.** It is stable to within
  0.3 percentage points across four horizons, which is a strong constraint on
  any candidate, and nothing more is claimed.
* **One year is not a climate.** Both models still carry the same deep ocean
  from the shared restart.
* **Every band is unsaturated at day 360.** The `u!` materiality test says only
  P1's `no` survives that fact unaided; the other `gap-at-N×` rows are upper
  bounds.
* **n = 4.** Every floor carries ~41 % relative standard error.

## 9. Controls, and the three that fired

All eleven registered controls are green in the shipped run. **Three fired on
real defects during development**, each of which would otherwise have reached
this document.

| control | what happened |
|---|---|
| **K3b dry-row census** | found row 198 (north wall) **fully dry**, mirroring row 0. The two domain walls are now a NAMED exception with a MEASURED reason — **K3c** poisons both entire wall rows with `u = 1e6` and requires every reported statistic to move by **exactly 0.0** — and any new dry row is a refusal rather than a widening. |
| **K5 dry-cell plant** | fired at **2.132e−14 Sv**, which was **not** a mask leak. The NEMO loader returns a `moveaxis` view and `.copy()` makes it C-contiguous, so `einsum` sums in a different order and the *same values* give a different band transport. Both arms now come from one contiguous baseline, and **the layout sensitivity is measured and printed** — because 2e−14 Sv is exactly what a sloppy A/B would manufacture. |
| **jet-profile empty-level refusal** | aborted on a level with no wet cell. DINO's tropical bathymetry does not reach the deepest model level, so the profile is **truncated at the deepest wet level and the truncation is printed**. The refusal is kept for the two cases that *are* defects: no wet level at all, and wet levels that are not a contiguous prefix from the surface. |

**K9, the known-answer control**, reproduces the verdict run's own recorded
day-360 south-of-band gap of **−0.95191 Sv to 2.2e−06 Sv**, compared **in
code**, and **aborts on a planted wrong recorded value** (shown non-vacuous in
the self-test).
**K5/K6** are the non-vacuity pair: the dry plant moves every band by exactly
0.0, the same plant on a wet cell inside P4 moves P4 by 1.4e+05 Sv and P1 by
exactly 0.0.
**K7b** (the six bands summing to the full section, 2.8e−14 Sv over all eight
runs) is **an identity in the row index** and is labelled as such — it is kept to
catch a mis-sliced band, not because physics can break it.

## 10. Retractions and in-place corrections

Three, all caught inside this lane before publication.

1. **"legoESM's equatorial westward surface jet is ~10 % weak" — INCOMPLETE, and
   corrected in place.** The first jet table sampled every third level and
   stepped straight over the trough at level 2. The difference is a **dipole**
   (+0.0445 m/s at 5 m, −0.0720 m/s at 26 m where NEMO reads only −0.0308), not
   a uniform weakening. The surface number is right; the description it invited
   was wrong.
2. **"the equatorial disagreement reaches 234 % at depth" — WITHDRAWN.** That
   line divided by NEMO's own value on a profile that **crosses zero**, so it
   reported the crossings, not the disagreement. It now normalises by the
   profile's own surface magnitude, one fixed nonzero number per profile, the
   same on both sides.
3. **The first F4 colour scale — WITHDRAWN.** It was set by the field's own
   maximum, a single equatorial cell at 0.472 K against a 99.5th percentile of
   0.068 K, so every band except the equator rendered white. The figure now
   draws two panels with both numbers in its title.

Nothing from an upstream document is retracted by this audit. The verdict run's
−0.0385 Sv "north of band" figure is **reproduced exactly**; what changes is what
it may be attributed to.

## 11. What the next lane should do first

1. **The per-row walk across rows 40–90** — the one measurement that decides
   whether P3 is its own object or the channel's flank. Offline, minutes.
2. **The equatorial dipole's owner.** A profile is not a term; the next step is a
   term-by-term near-surface momentum comparison on the equator row from the
   recorded states, not another statistic on `u`.
3. **Do not tune the equatorial transport.** It is already at 0.98× its floor
   while the profile behind it is at 166×. Optimising the number that is already
   right would move the one that is wrong in an unconstrained direction — the
   same trap the campaign recorded for lateral viscosity and channel transport.
