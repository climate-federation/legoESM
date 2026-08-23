# The 3-D T/S divergence atlas of the DINO verdict year

Instrument: `scripts/validate/ocean_fidelity/dino_1226/ts_divergence_atlas.py`
Pre-registration: `scripts/validate/ocean_fidelity/dino_1226/PREREG_ts_divergence_atlas.md` (7e088914e, before any statistic existed)
Upstream result this follows: `docs/ocean/fidelity/dino_verdict360_result.md` (a1387f1f7)

The verdict run answered *whether* the shipped DINO card matches NEMO over a
year from a shared restart, in transport terms: yes in the circumpolar channel
(17 ppm), no in the southern basin (-0.95 Sv). It did not say where the two
models' **water** differs. This does.

Everything here is offline from recorded states. No model was stepped.

**This document was rewritten after two independent adversarial reviews.** The
first draft's headline claims were cut down by controls the reviews demanded and
the instrument now runs. Section 7 lists every retraction; read it before citing
anything above it.

---

## 1. Inventory, and what was refused

| side | artifact | horizons carried | fields | provenance |
|---|---|---|---|---|
| legoESM | `/tmp/dino_verdict360/m{0..3}_*.npz` | **37**: days 0,10,...,360 | `T3d`, `S3d`, `u3d`, `eta3d` | one launch sha `a7b940f75` across all four; fp64 run, fp32 storage; vertical ladder `both`; seasonal clock 15552000 s = day 180.00 |
| NEMO | `.../DINO/RUN_VERDICT360_M{0..3}/DINO_<kt>_restart_*.nc` | **20**: 0,10,30,60,90,120,150,180,210,240,270,280,290,...,360 | `tn`, `sn`, `un` | certified binary; member 3 resolves to `/tmp/dino_v360_m3` after the home quota filled mid-campaign |

**Scored horizons = the 20-horizon intersection.** A horizon present on one side
only is dropped, never interpolated.

**Deliberately NOT used: the `floor90` artifacts** (`/tmp/dino_floor90`). The
task brief expected them to supply extra day-30/60 states; they are not needed,
because the verdict members already carry days 30 and 60 on both sides, and
`floor90` is a *different run*. Splicing it into a growth curve would be a
protocol difference — a confound — for horizons that cost nothing to get from
the run under test. The same reasoning excludes every other `/tmp/dino_*` arm.

### Controls, all green before any number below was read

| control | result |
|---|---|
| C1 provenance | one sha, four members, all stamps present, all 20 NEMO horizons on disk. The vertical ladder is refused on **value** (`both`), not merely printed; a member outside the certified NEMO tree is **refused** unless it is in a named allow-list with a reason — member 3 is the one named exception |
| C1b seasonal clock | all four members forced at **180.00 d** of the 360-day year; the restart is at **180.00 d** |
| C2 the kick is real | NEMO members 1-3 differ from member 0 at day 0 by 4.6-5.2e-14 relative on `tn`, on the wet mask |
| C3 day-0 identity | max\|dT\| **9.5e-07 K**, max\|dS\| **1.9e-06 g/kg**, bar 1e-05 (the fp32 storage quantum) |
| C4 NaN | zero non-finite values on the wet mask; the difference field is zeroed off the mask so a dry-cell NaN cannot reach a sum; every empty selection **aborts** rather than returning NaN |
| C5 planted violation | a **dry** cell given a difference of 1e6 moves **all ten** reported reductions by exactly 0.0; the same plant on a **wet** cell moves the global rms 1.9e-07 → 2.5e+02, so the check is not vacuous |
| C6 dtype | every comparison array, every difference field and every weight fp64 |
| C7a linearity | the per-row split sums to the group total — **an algebraic identity that cannot fail on physics**, kept only to catch a future non-linear refactor, and labelled as such in the code |
| C7b **known answer** | the day-360 south-of-band gap this instrument computes is **-0.95191 Sv** against the **-0.95191 Sv** recorded in a1387f1f7, agreeing to **2.2e-06 Sv**, compared **in code**. Shown non-vacuous: planting -0.90000 as the recorded value aborts the run |

Mask: `mesh_mask.tmask` AND the legoESM `land_mask`, the same 3-D mask on both
sides, 342134 wet T-cells.

Weights: **volume, `e1t * e2t * e3t_0`** (partial cell), zero on land. Every rms
is `sqrt(sum V d^2 / sum V)`; every *share* re-weights each row or box by its own
volume. There is no unweighted mean over cells of unequal volume anywhere in the
instrument, and the self-test proves the weighting is load-bearing by excluding
the unweighted answer on a case with an analytic result. Note the weight is the
*reference* partial-cell thickness, not the model's live free-surface thickness —
identical on both sides, so every share and every ratio is valid, but the total
2.534e17 m³ is not the model's live volume.

Depth classes are **fixed geometric bands shared by both models** — upper
< 200 m, interior 200-1400 m, abyss > 1400 m — *not* a diagnosed mixed layer,
because a model-diagnosed mixed-layer depth would differ between the two models
and be a protocol difference. The registration's parenthetical level counts
(13/15/8) are wrong; the rule is the depth threshold and it gives **13/14/9**.

---

## 2. The growth curves, and why "saturates" is the wrong word

Volume-weighted rms of (legoESM - NEMO). "floor" is the chaotic floor: the RSS of
each side's own member-vs-control rms, the verdict run's convention.

### Temperature [K]

| day | ALL | floor | ALL/floor | south basin | channel | north basin | upper | interior | abyss |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.86e-07 | 6.8e-14 | — | 1.15e-07 | 1.44e-07 | 1.91e-07 | 4.31e-07 | 2.09e-07 | 1.38e-07 |
| 10 | 5.38e-03 | 1.4e-09 | — | 1.86e-03 | 1.12e-03 | 5.72e-03 | 2.40e-02 | 7.80e-04 | 1.22e-04 |
| 90 | 1.156e-02 | 1.0e-04 | 113 | 3.04e-03 | 2.52e-03 | 1.230e-02 | 5.15e-02 | 1.75e-03 | 3.85e-04 |
| 180 | 1.211e-02 | 4.7e-04 | 26 | 3.47e-03 | 3.42e-03 | 1.287e-02 | 5.39e-02 | 2.04e-03 | 4.49e-04 |
| 270 | 1.232e-02 | 4.1e-04 | 30 | 3.18e-03 | 4.17e-03 | 1.306e-02 | 5.48e-02 | 2.35e-03 | 4.32e-04 |
| 360 | **1.274e-02** | 6.6e-04 | **19** | 4.23e-03 | 4.20e-03 | 1.351e-02 | 5.67e-02 | 2.42e-03 | 5.58e-04 |

### Salinity [g/kg]

| day | ALL | floor | ALL/floor | south basin | channel | north basin | upper | interior | abyss |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 3.27e-04 | 1.2e-09 | — | 3.24e-04 | 1.43e-04 | 3.42e-04 | 1.44e-03 | 1.09e-04 | 2.13e-05 |
| 90 | 8.24e-04 | 8.4e-06 | 98 | 4.38e-04 | 3.34e-04 | 8.69e-04 | 3.64e-03 | 2.24e-04 | 5.72e-05 |
| 180 | 8.13e-04 | 8.1e-05 | 10 | 3.51e-04 | 3.23e-04 | 8.59e-04 | 3.59e-03 | 2.38e-04 | 4.93e-05 |
| 270 | 8.01e-04 | 4.9e-05 | 16 | 3.05e-04 | 3.35e-04 | 8.46e-04 | 3.53e-03 | 2.57e-04 | 4.14e-05 |
| 360 | **7.85e-04** | 5.9e-05 | **13** | 2.58e-04 | 3.21e-04 | 8.30e-04 | 3.46e-03 | 2.50e-04 | 3.70e-05 |

Ratios before ~day 90 are omitted: the denominator is a storage artifact until
the 1e-14 kick has grown.

### The amplitude settles; the field does not

**RETRACTED from the first draft: "the global tracer divergence saturates".** A
flat rms is a statement about **amplitude only** — a difference field completely
re-drawn between two horizons at constant amplitude has a flat rms too. The
instrument now computes the discriminating statistic, the norm of the *change* in
the difference field:

| field | rms at day 360 | rms of (d360 − d90) | ratio to the field's own rms |
|---|---|---|---|
| T | 1.274e-02 K | **1.074e-02 K** | **0.84** |
| S | 7.85e-04 g/kg | **8.96e-04 g/kg** | **1.14** |

Between day 90 and day 360 the temperature difference field changes by 84 % of
its own size and the salinity difference field by 114 %, while the amplitudes
move by 10 % and −5 %. Per scored interval the change in the field runs 8× to
2400× the change in the amplitude.

**CONFIRMED, corrected statement:** the *amplitude* of the tracer difference
settles by day 90; the *pattern* does not — by day 360 it is substantially a
different field at the same size.

### The contrast with the transport gap, on matched windows

**RETRACTED: the first draft set a 270-day tracer window against a 90-day
transport window**, anchored at day 270, which is the *minimum* of a non-monotone
trajectory the upstream document warns about. On matched windows:

| window | T global rms | T southern-basin rms | \|south-of-band gap\| |
|---|---|---|---|
| 90 → 360 | ×1.10 | ×1.39 | ×2.24 |
| 180 → 360 | ×1.05 | ×1.22 | ×4.10 |
| 270 → 360 | ×1.03 | ×1.33 | ×16.6 |

The contrast is real on every matched window, but at 90→360 it is a factor of
1.4-2.2, not "flat versus ×16". And in the basin that owns the gap the tracer rms
is **not** flat: it grows 39 % over the same 270 days.

---

## 3. Where the divergence is — and how much of that is the floor

Share of the total volume-weighted squared difference, day 360:

| | south/upper | south/int | south/abyss | chan/upper | chan/int | chan/abyss | **north/upper** | north/int | north/abyss |
|---|---|---|---|---|---|---|---|---|---|
| T gap | 0.0001 | 0.0007 | 0.0010 | 0.0105 | 0.0010 | 0.0001 | **0.9776** | 0.0089 | 0.0002 |
| T floor (legoESM vs itself) | 0.0012 | 0.0073 | 0.0294 | 0.0035 | 0.0701 | 0.0014 | **0.7995** | 0.0873 | 0.0004 |
| T floor (NEMO vs itself) | 0.0002 | 0.0028 | 0.0127 | 0.0021 | 0.0393 | 0.0001 | **0.9334** | 0.0094 | 0.0001 |

The raw gap is 97.8 % in one box (95.4 % for salinity) — but so is **80 % of
legoESM's difference from itself and 93 % of NEMO's**. The identification of the
box is solid: its volume-weighted centroid sits at latitude **+1° to +6°** and
depth **27 m → 59 m**, the equatorial upper ocean and thermocline, and the
zonal-mean panels show a sharp equatorial dipole deepening through the year. But
"98 % of the divergence lives there" is, to first order, a statement about where
**any** perturbation to this configuration grows.

Normalising each box by its own floor answers the different question — where do
the two models differ by more than one of them differs from itself:

| T, signal-to-noise share, day 360 | south/upper | chan/upper | chan/abyss | north/upper | north/abyss |
|---|---|---|---|---|---|
| share | 0.029 | **0.586** | 0.036 | 0.159 | 0.122 |

(Only the saturated horizon is readable; before the kick has grown the
denominator is a storage artifact.)

**CONFIRMED, corrected statement:** by raw amplitude the tracer difference is
overwhelmingly equatorial-upper, and so is the chaotic floor. Measured against
that floor the leading box is the **channel upper ocean** (0.59 of the
signal-to-noise share), not the equator. The southern basin holds **0.18 %** of
the raw temperature difference — not the 0.02 % the first draft stated.

The claim that the tracer divergence and the momentum deficit have different
primary geographies survives on both measures — neither puts the maximum in the
southern basin — but it must be stated from the normalised measure, because the
raw one is reproduced by the floor.

---

## 4. Question 2 — first departure

Registered statistics: volume-weighted pattern correlation of `|d|` against day
10; the `d^2` centroid; and **day-10 hotspot retention** — the share of the total
`V d^2` at horizon *t* inside the cells holding the top 5 % of the wet volume
when ranked by `|d|` at day 10. Null = 0.05, verified by a random-volume control
and by an unequal-weight self-test.

Registered bars: GROWS IN PLACE if r(day10, day360) ≥ 0.50 **and**
retention(360) ≥ 0.25. SPREADS if retention(360) < 0.15 **and** the centroid
moves > 10 rows or > 500 m. Otherwise MIXED.

### Registered, whole domain, day 360

| field | r(\|d\|,\|d10\|) | retention fwd | retention rev | self-concentration | centroid move |
|---|---|---|---|---|---|
| T | **0.492** (bar 0.50) | **0.971** | 0.978 | 0.997 | 4.4 rows, +31 m |
| S | **0.425** | **0.906** | 0.943 | 0.994 | 0.6 rows, +14 m |

**Registered outcome: MIXED for both fields. The bar was not moved.** The spread
arm is refuted by 19× on retention; the grow-in-place arm misses on pattern
correlation by 0.008 (T). The two statistics disagreeing is the result: r decays
from 1.00 to 0.49 while retention holds at 0.97 — the hotspot *region* is fixed
from day 10 and the pattern *inside* it turns over, which is the same conclusion
section 2's increment norm reaches by an independent route.

### Post-hoc, per region — NOT registered, labelled post-hoc throughout

Added after the share table showed one box dominating the registered numbers.

| field / region | r(360) | ret fwd | ret rev | self | centroid, day 10 → day 360 |
|---|---|---|---|---|---|
| T north basin | 0.494 | 0.973 | 0.981 | 0.998 | +6.0° → +2.0°, 24 → 55 m |
| T channel | 0.281 | 0.832 | 0.591 | 0.962 | 311 → 173 m |
| **T south basin** | **0.036** | **0.054** | **0.282** | **0.842** | **-65.54° → -68.39°, 1081 → 1406 m** |
| S north basin | 0.430 | 0.910 | 0.964 | 0.995 | 38 → 73 m |
| S channel | 0.314 | 0.793 | 0.467 | 0.940 | 459 → 223 m |
| **S south basin** | **0.087** | **0.054** | **0.326** | **0.907** | **1124 → 1730 m** |

**CONFIRMED: the southern basin behaves oppositely to the rest of the domain.**
Its forward retention at day 360 is 0.054 against a 0.05 null — the day-10
hotspot has been abandoned — and this is *not* a diffuseness artifact, because
the day-360 field is itself highly concentrated (self-retention 0.842 for T,
0.907 for S). Its
divergence deepens through the year (T +325 m, S +606 m) and, for temperature,
migrates 2.85° south toward the wall, while the rest of the domain's stays
pinned where it appeared in the first ten days.

**Softened from the first draft: "no memory of where it was at day 10".** That
was one-directional. Reverse retention — how much of the *day-10* squared
difference sits in the *day-360* hotspot — is 0.282, five times the null, but it
is unstable across horizons (0.003 at days 270-300, 0.29 at days 310/350/360), so
"the day-360 hotspot was already enriched at day 10" is true at the endpoint and
false three horizons earlier. What is stable, and is the finding: **the forward
retention sits at the null from day 210 onward**, so the day-10 hotspot is
abandoned and does not come back.

---

## 5. Question 3 — the feedback fingerprint

The momentum-deficit geography is **measured here**, not quoted: per-row zonal
transport deficit from `acc_driver_decomp.group_transport`, `A.umask` on both
sides, the committed lons-2..-2 reducer, southern-basin rows 1-13. Control C7b
confirms these rows reproduce the campaign's recorded south-of-band gap.

The campaign's committed wall split is used verbatim: **wall rows 1-5**, main
rows 6-13. The task brief said "the 4 wall rows"; the committed split is five and
was not changed to match prose.

### F1 co-location — and why the registered bar is not usable here

**RETRACTED: "temperature is COUPLED to the momentum deficit over the final
quarter".** The registered bar |ρ| ≥ 0.56 is the p<0.05 critical value for
**thirteen independent samples**. These thirteen are adjacent latitude rows in
one basin and are strongly autocorrelated. Three controls, all now in the probe:

| day | ρ(T,\|D\|) | exact cyclic-shift p | n_eff | correct \|ρ\| critical | rows below 2× their own floor | wrong-time rank |
|---|---|---|---|---|---|---|
| 10 | **-0.786** | **0.077** | 13.1 | 0.514 | 0 of 13 | **19/19** |
| 90 | 0.077 | 0.923 | 13.8 | 0.503 | 0 | 12/19 |
| 240 | 0.473 | 0.231 | 8.7 | 0.610 | **8** | 15/19 |
| 300 | 0.786 | 0.077 | 6.2 | 0.697 | 4 | 1/19 |
| **360** | **0.736** | **0.231** | **4.4** | **0.793** | 1 | **7/19** |

- The **exact cyclic-shift null** — roll the deficit profile by 1..12 rows, which
  preserves both profiles' shapes and destroys only their alignment — puts **no
  horizon after day 150 below p = 0.077**, and that value is the test's
  resolution floor (1/13), not a significance level. Day 360 sits at p = 0.231.
- The effective sample size falls to **4.4** by day 360, moving the correct
  critical |ρ| to **0.793**, above the observed 0.736.
- The **wrong-time null** is decisive: the day-360 tracer profile matches the
  day-360 deficit *worse* than six other horizons' deficits (rank 7 of 19). There
  is no time-specific relationship.
- At days 210-290, **8 to 10 of the 13 rows** carry a deficit below twice their
  own ensemble floor, so their ranks are set by chaotic noise.

**What survives, CONFIRMED:** the **day-10 anti-alignment**. ρ = -0.786 sits at
the shift null's resolution floor and ranks **19 of 19** in the wrong-time null —
the single most extreme pairing in the year, and the only time-specific result in
F1. The earliest tracer departure is where the transport deficit is *not*.

**What replaces the "coupling" claim, and needs no correlation at all:** by the
final quarter both the temperature divergence and the transport deficit are
concentrated in the wall rows. That is F3 below, stated directly.

**RETRACTED: "salinity is NOT coupled".** Applied as registered — two-sided, at
every horizon — salinity **clears the bar at days 10 (-0.769), 90 (-0.775) and
120 (-0.775)**, all negative, and never after day 150. The first draft admitted
temperature's negative day-10 value as a finding while excluding salinity's with
an "after day 150" qualifier that appears nowhere in the registration, and
omitted day 120 from the table. That was the same criterion applied two ways.
Salinity's shift-null p at days 90 and 120 is 0.077, the same resolution floor as
temperature's day 10.

### F2 lead/lag

**RETRACTED: "the argmax favours tracer-leads-momentum by about ten days" — the
direction was inverted and the shape was an artifact.**

The registration's legend says k<0 means the tracer leads. It does not: the
deficit is read at the *later* horizon for k>0, so a peak at k<0 means the tracer
matches an *earlier* deficit, i.e. **momentum leads the tracer**. The formula is
the registered one; its legend was wrong, and the probe now prints the correction.

On **one common set of 15 horizons** (the first draft averaged each offset over a
different subset, and two offsets paired against the day-0 deficit, which is
roundoff):

| k | median dt [d] | mean r (T) | sd | r(k) − r(0), paired ± SE |
|---|---|---|---|---|
| -2 | -50 | 0.668 | 0.454 | -0.087 ± 0.042 |
| **-1** | -30 | **0.779** | 0.252 | **+0.023 ± 0.042** |
| 0 | 0 | 0.755 | 0.349 | — |
| +1 | +10 | 0.735 | 0.416 | -0.021 ± 0.026 |
| +2 | +20 | 0.726 | 0.458 | -0.029 ± 0.035 |

k = -2 is now the **minimum**, so the "monotone decline from the peak" the first
draft described does not exist on a controlled sample. The paired difference
between the argmax and zero offset is **+0.023 ± 0.042** — 0.55 standard errors.

**CONFIRMED: no lead or lag is resolved, quantitatively.** The offset is
indistinguishable from zero, and the error bar the first draft said could not be
computed is one line of code.

### F3 wall-versus-main share

Tracer share = share of the southern-basin volume-weighted `d^2` in wall rows
1-5, **each row re-weighted by its own wet volume**. The first draft summed
per-row mean squares unweighted — the reduction this campaign has a standing
retraction for — and every number in that table was wrong by up to 0.14.

| day | T wall share | S wall share | \|D\| wall share |
|---|---|---|---|
| 10 | 0.033 | 0.038 | 0.743 |
| 90 | 0.334 | 0.034 | 0.760 |
| 180 | 0.620 | 0.117 | 0.778 |
| 360 | **0.741** | **0.202** | 0.770 |
| time-weighted year mean | 0.479 | 0.101 | 0.722 |

**The geometric null is the wall rows' share of the basin's wet volume: 0.286** —
not the 0.385 share by row count the first draft used.

- **CONFIRMED: temperature migrates into the wall rows** — from 0.033 (nine times
  under-represented against the volume null) at day 10 to 0.741 (2.6×
  over-represented) at day 360.
- **CONFIRMED: salinity avoids them** — 0.202 at day 360, below the 0.286 volume
  null, and below half the deficit's share, which is the registered criterion.
  The corrected weighting moved this number in salinity's favour; the first
  draft's 0.338 sat *above* the geometric null.
- **RETRACTED: "the deficit's wall share is roughly constant at 0.72-0.78".** Over
  the 19 scored horizons it ranges **0.454 to 0.851** (mean 0.731). The first
  draft quoted the horizons that happen to be flat.

---

## 6. Seasonal confound — applies to every statement about *when*

The restart is day 180 of a 360-day year, so **elapsed time and season are
degenerate in this run**. The "final quarter" (days 270-360) is day-of-year
90-180, one particular season; the early horizons (days 10-60) are day-of-year
190-240, another. Every sign change and every "grows in the final quarter"
statement above is confounded with season and **cannot be deconfounded from a
single year**. The only same-season pair in the run is day 0 and day 360, and day
0 is the identity, so no seasonal control exists inside this run at all.

The upstream verdict document carries the same limitation and uses a full-cycle
mean as its remedy. The first draft did not mention season anywhere; that was a
regression against the document it builds on.

The horizon cadence compounds it: 30-day for the first three quarters, 10-day for
the last, so half the scored horizons sit in the final quarter and any unweighted
mean over horizons is mostly a mean of that quarter. The F3 year mean is
time-weighted for this reason.

---

## 7. What survives, and every retraction

**CONFIRMED (controls green, instrument reproduces a recorded number to 2.2e-06 Sv):**

1. The *amplitude* of the tracer difference settles by day 90; the *pattern* does
   not. The day-90-to-day-360 change in the temperature difference field is 0.84×
   the field's own size, and 1.14× for salinity.
2. The raw tracer difference is ~98 % equatorial-upper — and so is 80-93 % of
   each model's own ensemble spread. Measured against that spread, the leading
   box is the channel upper ocean (0.59 of the signal-to-noise share).
3. Outside the southern basin the divergence keeps its day-10 hotspot (forward
   retention 0.83-0.97 against a 0.05 null) while the pattern inside it turns
   over.
4. Inside the southern basin the day-10 hotspot is abandoned from day 210 onward
   (forward retention 0.054, at the null) without the field becoming diffuse
   (self-concentration 0.842 for T, 0.907 for S); the divergence deepens and, for temperature,
   migrates 2.85° toward the southern wall.
5. Temperature's wall-row share rises from 0.033 to 0.741 against a 0.286
   geometric null; salinity ends at 0.202, below it. The deficit's own wall share
   ranges 0.454-0.851 all year.
6. The **day-10 anti-alignment** between the temperature divergence and the
   transport deficit (ρ = -0.786, rank 19/19 in the wrong-time null) is the one
   time-specific relationship in F1.

**RETRACTED, in full:**

| # | first-draft claim | what the controls show |
|---|---|---|
| R1 | "the global tracer divergence saturates" | the amplitude does; the field is 84 % (T) / 114 % (S) re-drawn between days 90 and 360 |
| R2 | "flat tracer rms versus a compounding transport gap" | windows were mismatched; on matched windows the contrast is ×1.4-2.2, and the southern-basin tracer rms grows 39 % |
| R3 | "the southern basin holds 0.02 % of the temperature divergence" | it holds **0.18 %** (0.0001 + 0.0007 + 0.0010 at the printed precision) |
| R4 | "temperature is COUPLED over the final quarter" | fails the exact cyclic-shift null (p = 0.231 at day 360), the effective-sample-size correction (critical 0.793 vs observed 0.736) and the wrong-time null (rank 7/19) |
| R5 | "salinity is NOT coupled" | applied as registered it clears the bar at days 10, 90 and 120, all negative; the first draft used a qualifier absent from the registration and omitted day 120 |
| R6 | "the argmax favours tracer-leads-momentum by ~10 days, declining monotonically" | the sign convention was inverted (a k<0 peak means momentum leads); on a common sample k=-2 is the minimum, and the paired offset is +0.023 ± 0.042 |
| R7 | the T/S wall shares (0.7565, 0.2011, 0.8422, 0.3383 …) | computed by an unweighted sum of per-row mean squares; every value wrong by up to 0.14 absolute |
| R8 | "the deficit's wall share is roughly constant at 0.72-0.78" | it ranges 0.454-0.851 |
| R9 | "no memory of where it was at day 10" (southern basin) | one-directional; reverse retention is 0.282 at day 360 but 0.003 three horizons earlier, so only the forward statement is stable |
| R10 | the C7 control "reproduces the recorded number" | the original C7 was an algebraic identity that could not fail and never compared against the recorded value. C7b now does, in code, and aborts on a planted wrong value |

**Not claimed, and now actively refuted rather than merely disclaimed:** any
coupling between the tracer divergence and the momentum deficit. The first draft
flagged the wall-heaviness of both profiles as a confound and reported the
correlation anyway; the controls show that confound accounts for the whole signal.

**Deviations from the pre-registration, all disclosed:**

* the stampless-artifact refusal was not applied literally — the missing clock
  stamp postdates the runs and cannot be added, so its substance was supplied
  from the oracle (control C1b);
* the hotspot set is ranked by intensity and cut by volume, where the
  registration says "top 5 % of V·d²" — ranking by V·d² would make the set a
  statement about cell size and destroy the exact 0.05 null;
* the centroid uses the 3-D partial-cell depth, where the registration says the
  1-D ladder;
* the per-region Question 2 statistics are POST-HOC;
* the registration's F2 sign legend is inverted and is corrected in the output;
* the registration's depth-class level counts are wrong (13/14/9, not 13/15/8).

**One-line follow-up, named and NOT started:** the southern basin's divergence
abandoning its day-10 hotspot while deepening and moving wall-ward is what a
term-by-term southern-basin *tracer* budget would resolve, alongside the
transport budget the verdict run already named.

## 8. Figures

Two multi-panel atlases, one per field, 3 × 5 panels: zonal-mean depth-latitude
difference at days 10/90/180/270/360; horizontal maps at 5 m (days 10, 90, 360)
and at 514 m and 2522 m (day 360); growth by region; growth by depth class; the
Question 2 statistics against their bars; the per-row Question 3 overlay; and the
centroid track for the whole domain and the southern basin separately.

Regenerate with:

```
JAX_ENABLE_X64=1 PYTHONPATH=<worktree>/packages/ocean:<worktree>/packages/core:<worktree>/src \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/ts_divergence_atlas.py --out-dir <dir>
```

which writes `ts_divergence_atlas_T.png`, `ts_divergence_atlas_S.png` and
`ts_divergence_atlas.npz`. The npz carries the per-horizon records; the post-hoc
per-region Question-2 statistics exist for every horizon after day 10 and are
absent at day 0, where they are undefined.
