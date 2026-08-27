# Pre-registration — the equatorial and northern REGIONAL audit of the DINO verdict year

Registered **before any regional statistic existed**. Nothing below was chosen
after seeing a number. Every deviation is disclosed in the result document.

Instrument: `scripts/validate/ocean_fidelity/dino_1226/regional_audit.py`
Upstream results this follows, and does not re-derive:
* `docs/ocean/fidelity/dino_verdict360_result.md` (a1387f1f7) — the transport verdict
* `docs/ocean/fidelity/dino_ts_divergence_atlas.md` (c754652a4) — the T/S atlas and its ten retractions
* `docs/ocean/fidelity/dino_campaign_synthesis.md` — the instrument canon

Everything is **offline from recorded states**. No model is stepped. CPU only.

---

## 1. The question

The verdict run split the DINO section into three latitude groups — south of the
channel, the channel, north of it — and found the model right in the channel and
wrong in the southern basin. **"North of band" is 150 of 199 rows**, from 44.6°S
to 69.9°N. It is one number covering the entire tropics and the whole northern
hemisphere, and it has never been opened.

This audit opens it. Two specific sub-questions the campaign has never asked:

* **Q1** Does the equatorial ocean carry a transport gap of its own, and is it
  distinguishable from the two models' own run-to-run spread *measured in that
  band*?
* **Q2** The atlas found the raw tracer difference is ~98 % equatorial-upper but
  that the ensemble spread is too. Re-reduced into finer bands, does the equator
  hold its **own** divergence hotspots, or is the tropical divergence a pattern
  advected/inherited from elsewhere?

## 2. The bands — registered as ABSOLUTE T-ROW INDICES

The mesh is fixed (`mesh_mask.nc`, 199 T-rows, Mercator, `gphit` symmetric about
row 99 which sits at **exactly** 0.000000°). Bands are registered as row indices
so no later re-reading of a latitude threshold can move them.

**The six-band PARTITION** (disjoint, exhaustive, sums to all 199 rows):

| # | band | T-rows | latitudes (col 25) |
|---|---|---|---|
| P1 | south of band | 0–13 | south wall … −64.44° |
| P2 | channel band (re-entrant) | 14–48 | −64.44° … −45.35° |
| P3 | southern subtropics | 49–78 | −44.65° … −20.55° |
| P4 | equatorial ±20° | 79–119 | −19.61° … +19.61° |
| P5 | northern subtropics | 120–150 | +20.55° … +45.35° |
| P6 | northern subpolar | 151–198 | +46.05° … +69.85° (north wall) |

**P1 and P2 are BIT-IDENTICAL to `acc_driver_decomp.LAT_GROUPS[0]` and `[1]`**
(`slice(0, A.J0)`, `slice(A.J0, A.J1+1)`), imported, never re-typed. P3+P4+P5+P6
is exactly `LAT_GROUPS[2]` ("north of band"), so this audit is a *refinement* of
the recorded three-group split and must reproduce it by summation.

**One NESTED band, reported alongside P4 and never summed with it:**

| # | band | T-rows | latitudes |
|---|---|---|---|
| E10 | equatorial ±10° | 89–109 | −9.95° … +9.95° |

Both ±10° and ±20° are reported because neither is the honest equatorial width
on its own: ±10° is roughly the wind-driven tropical cell, ±20° captures the
subtropical cells' equatorial flanks, and DINO's own analytic forcing carries an
equatorial dip of Gaussian width 7.5° (`usrdef_sbc`, recorded in
`dino_l1_exactness_audit.md`). Reporting one width would be a choice presented
as a fact.

**Depth classes** are the atlas's, imported unchanged (`ts_divergence_atlas.DEPTH_CLASSES`):
upper < 200 m, interior 200–1400 m, abyss > 1400 m — fixed geometric bands
shared by both models, never a model-diagnosed mixed layer.

## 3. Horizons

**Registered primary horizons: days 90, 180, 270, 360.** These are the verdict
run's own registered horizons; the verdict table below is scored at these four
and nowhere else.

Day 0 and day 10 are carried as **controls only** (day-0 identity; day-10
pre-growth), never scored — the kick has not grown and the denominator is the
fp32 storage quantum.

The remaining horizons of the 20-horizon lego∩NEMO intersection (0, 10, 30, 60,
90, 120, 150, 180, 210, 240, 270, 280…360 by 10) are carried as **trajectory
context and are labelled POST-HOC** wherever used. They cannot promote or demote
a registered verdict.

## 4. Reductions — every one imported, none re-typed

| quantity | function | notes |
|---|---|---|
| band zonal transport [Sv per lon] | `acc_driver_decomp.group_transport(u, A.umask, rows)` | e3t_1d weighting, `A.e2u_col` widths — verbatim the recorded `acc_full` integrand restricted to rows |
| the scalar per band | `acc_driver_decomp._avg` (mean over lon, ends 2 trimmed) | the verdict table's **mean** reduction; the median reduction is NOT additive over bands and is not used |
| barotropic / baroclinic split | `acc_driver_decomp.section_bt_bc(u, A.umask, rows=…)` | `bt = u_bottom·H` (reference-level), `bc = ∫(u−u_bottom)dz`. **bt+bc ≡ section total is asserted per band**, not assumed |
| T/S divergence | `ts_divergence_atlas.wrms(d, w, sel)` | volume-weighted, `w = e1t·e2t·e3t_0` zeroed off the shared wet mask |
| pattern correlation | `ts_divergence_atlas.wcorr(a, b, w)` | volume-weighted Pearson on the difference fields |
| ensemble spread | `floor90_ensemble.spread` | (max-pairwise, sample std); **std is primary**, NaN propagates |

**The wet mask is the atlas's and the gate's: `A.tmask` AND the legoESM
`land_mask`, the same 3-D mask on both sides.** A per-model mask is a protocol
difference, i.e. a confound.

**Weighting law, stated because the campaign has been burned by it** (synthesis
§4, "layer-versus-thickness weighting"): every volume reduction is
**thickness-weighted by `e3t_0`**, never layer-averaged. A layer average
over-weights the 10 m surface layer ~54× against the 545 m bottom and has already
inverted the sign of three published wall rows. No unweighted mean over cells of
unequal volume appears anywhere in this instrument.

## 5. Floors — measured PER BAND, never borrowed

The verdict run's own rule, applied per band rather than globally:

```
floor_band(day) = sqrt( spread_lego_band(day)^2 + spread_NEMO_band(day)^2 )
```

where each side's spread is the **sample std (ddof=1) over its own 4 members**
(control + 3 seeds perturbed by a 1e-14 relative temperature nudge) of the same
band's statistic at the same day. **n = 4 per side.** A floor at n=4 carries
~41 % relative standard error and is a factor-of-two estimate; this is stated
next to every band's floor and is not re-argued per row.

**A global floor is NEVER transferred to a band.** The synthesis's own rule —
"the bar belongs to the statistic" — cost the campaign a wholesale re-score when
a saturated 10-year floor was applied to an unsaturated 90-day comparison.

## 6. Verdict bars — registered constants

Imported from `verdict360.py`, not re-typed:

| constant | value | meaning |
|---|---|---|
| `K_PREREG` | 2.0 | **INDISTINGUISHABLE ⟺ \|gap\| ≤ 2 × floor.** The registered rule; it decides. |
| `K_WELCH` | 2.45 | An estimated floor at ν≈6 needs 2.45, not 2.0. Ratios in [2.0, 2.45) are printed `no` by the registered rule while **not** being statistically resolved. Printed as its own column, never used to overturn. |
| `QUANTUM_MARGIN` | 10.0 | A band whose legoESM spread is under 10× the fp32 storage quantum of that band's own statistic is measuring the npz dtype. Verdict → **UNMEASURABLE**. |
| `SATURATION_RATIO_MAX` | 1.3 | over `SATURATION_QUARTERS` = (180→270) and (270→360) |
| `ONE_SIDED_DECADES` | 1.0 | the two sides' spreads within one order of magnitude, else the RSS is one model's dispersion wearing a two-sided name |

**Verdict vocabulary, fixed here:**

* **INDISTINGUISHABLE** — `|gap| ≤ 2 × floor`, and the floor is measurable
  (over the quantum margin) and saturated.
* **gap-at-N×** — `|gap| > 2 × floor`; N is `|gap|/floor` printed to 2 s.f.
* **UNMEASURABLE** — the band's legoESM spread is at or under `QUANTUM_MARGIN ×`
  its own fp32 quantum. **No verdict either way**; the gap is printed and
  carries no multiple. This is not a pass.
* **`u` flag** — the band's ensemble is **UNSATURATED** (the spread is still
  growing over the last two quarters). Asymmetric, exactly as registered
  upstream: an INDISTINGUISHABLE still stands (the runs did not separate even
  with room to), a `gap-at-N×` does **not** — the denominator is still growing,
  so N is an upper bound on the true multiple and the row reads
  `gap-at-N× (u)`.
* **`1s` flag** — the two sides' spreads differ by more than one decade, so the
  RSS floor is numerically one model's own dispersion.

## 7. The equatorial instrument blind spots, and how this probe designs around them

Registered here so the design cannot be rationalised afterwards.

1. **`f = 0` at row 99, exactly.** The campaign has already withdrawn one
   published enrichment (3.51×) that existed *only* because the equator row sat
   in a denominator where the vorticity flux structurally vanishes
   (`dino_wall_ldf_alignment.md`). **This probe computes NO vorticity-projection,
   no Coriolis-normalised and no f-weighted statistic anywhere.** A control
   asserts `f[99] == 0.0` and prints it, and every per-row normalised map this
   probe emits has its row-99 denominator printed so a reader can see it is not
   structurally zero.
2. **The structural-zero row is excluded from every normalised map, and the
   exclusion is PRINTED** — the row index, the reason, and the denominator it
   would have contributed.
3. **Heterogeneous-band masking** (synthesis §4): a dry row of exact zeros next
   to a wet value inverted a verdict 94 % → 37 %. Every band here is checked for
   its wet-cell count per row, and a band containing a fully dry row is refused
   rather than averaged. Row 0 is dry and is inside P1, which is why P1 is
   imported from the recorded slice rather than re-derived.
4. **Empty selection is FATAL.** A band or depth class that selects nothing
   aborts; it never returns NaN into a table.
5. **The median reduction is not additive over bands.** The verdict document
   says so explicitly about its own "ACC" row. This audit reports the **mean**
   reduction for every additive statement and never decomposes a median.

## 8. Controls — all must be green BEFORE any number is read

| id | control | pass condition |
|---|---|---|
| **K1** | provenance | one launch sha across the four legoESM members; ladder stamp `both` refused on value; NEMO member 3's off-tree directory is the one NAMED allow-list exception |
| **K2** | seasonal clock | all members at 180.00 d; supplied from the oracle restart per the atlas's C1b |
| **K3** | day-0 identity | max abs band transport difference at day 0 under the fp32 quantum |
| **K4** | NaN | zero non-finite in every band statistic; NaN is FATAL, never skipped |
| **K5** | **planted violation, dry cell** | a dry cell given `u = 1e6` moves **every** reported band transport by exactly 0.0 |
| **K6** | **planted violation, wet cell** | the same plant on a **wet** cell moves the containing band's transport by ≫ its floor — proves K5 is not vacuous |
| **K7** | **partition identity** | P1+…+P6 equals `acc_driver_decomp.group_transport` over all 199 rows to < 1e-9 Sv, and E10 ⊂ P4 |
| **K8** | **bt+bc identity** | per band, `bt + bc` equals the band section total to < 1e-9 Sv |
| **K9** | **known answer** | the day-360 P1 gap reproduces the recorded **−0.95191 Sv** to < 1e-4, compared IN CODE, and a planted wrong recorded value aborts the run |
| **K10** | dtype | every difference field, weight and accumulator fp64 |
| **K11** | structural zero | `f[row 99] == 0.0` printed; no reported statistic has it in a denominator |

**K5/K6 and K9 are the non-vacuity pair.** A control that cannot fire is the
defect this campaign has caught most often (synthesis §4, "cannot-fail
controls"), so each is shown to fail on a planted violation.

## 9. Predictions — registered so they can be wrong

| # | prediction | confirms | refutes |
|---|---|---|---|
| **R1** | The equatorial band (P4) carries a transport gap that is **larger than its own floor** at day 360 | gap/floor > 2 | ≤ 2 → the equator is indistinguishable and the tropics are clean |
| **R2** | P1 (south) remains the **largest** band gap in absolute Sv at day 360 | P1 is rank 1 | any other band exceeds it — a NEW regional finding |
| **R3** | The equatorial gap is predominantly **baroclinic** (|bc| > |bt| in P4) | yes | a barotropic equatorial gap is a different mechanism and a new finding |
| **R4** | Re-reduced per band, the equatorial T divergence **holds its own** hotspots — forward retention of its day-10 hotspot set stays above the 0.05 geometric null through day 360 | retention > 0.25 (the atlas's registered `Q2_RET_BAR`) | retention at the null → the tropical divergence is inherited, not local |
| **R5** | The northern subpolar band (P6) carries a **smaller** absolute gap than the northern subtropics (P5) | yes | no |

A prediction that fails is reported as failed, in the same table, with the
number. None of the five is load-bearing for a verdict; the verdicts come from
§6 applied to §5's measured floors.

## 10. Escalation rule

**Any band showing a gap greater than 5× its own measured floor is a NEW
REGIONAL FINDING and requires dual sign-off** — one code review of the
instrument and one physics review of the finding — before it is written up as
anything but a candidate. Registered here so the bar cannot be set after the
number is seen.

## 11. The headline sentence's honesty constraint

Whatever this audit finds, its headline carries the same three limits the
verdict run's does, verbatim in substance:

> This is a statement about **one year from a common ocean state**; it is not a
> statement that the two models share a climate, and the deep ocean has not
> adjusted on this timescale.

and it adds the one this audit owns:

> Every floor here is measured on **n = 4** members per side and is a
> factor-of-two estimate; a band verdict at 2–3× its floor is not resolved.
