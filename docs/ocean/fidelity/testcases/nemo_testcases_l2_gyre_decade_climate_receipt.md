# Receipt — GYRE decade climate tier: legoESM vs NEMO 5.0.2, ten years

Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_decade_climate.md`,
written and committed before either ten-year run existed. Every bar quoted
below is that document's; the scorer prints numbers and no verdict, and this
receipt is the interpretation.

Scoring tree: commit `044dce3d3`, clean. Scorer:
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_decade_climate.py`.
JSON: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/decade/decade_climate.json`.

## 0. What ran, and how it was admitted

| | legoESM | NEMO 5.0.2 |
|---|---|---|
| steps | 21600 (3600 days, 10 × 360-day years) | 21600 (`nn_itend`) |
| timestep | 14400 s, 6 steps/day | same, from the card's `namelist_cfg` |
| output | snapshot every 180 steps, 120 `day{NNN}.npz` | restart every 180 steps (`nn_stock`), 120 files |
| arithmetic | fp64 with libm transcendentals, `JAX_ENABLE_X64=1`, CPU | scalar-math build, the certified binary |
| wall | 15261 s | about 7 minutes |
| producing commit | `af374a04e`, clean tree | binary sha256 `a759e8b4…1960cd` |

The round brief specified 14400 steps and a 120-step restart cadence on an
assumed four steps per day. The card's own namelist sets `rn_Dt = 14400.`, so a
day is **six** steps; the decade is 21600 steps and a month is 180. The
namelist was taken as the authority and the acquisition script re-reads and
re-checks it at run time.

**Admission (operator's note AS).** The acquisition built nothing — it ran the
binary that already certified the from-rest year — and then compared its twelve
year-1 restarts against the un-instrumented round-132 daily record. Its log
reads `ADMISSION OK: all 12 year-1 months match round 132 byte for byte`,
followed by `all 120 monthly restarts present`. The decade is therefore the
same trajectory the campaign has been scoring, extended.

## 1. P1 — the preregistered GYRE surrogate for the FESOM2-JAX bar. **MET**, with room.

The FESOM2-JAX paper divides the JAX-minus-Fortran RMS by the
model-minus-observation bias. GYRE is idealised and has no observations, so the
preregistration fixed a surrogate denominator instead: the field's own RMS
about its area-weighted basin mean. This is *not* that paper's bar, and the
substitution matters in one direction — a field with strong basin-scale
gradients has a large denominator and will produce a small ratio. The
denominators are quoted in full below so a reader can apply their own.

Years 2-10 (months 13-120, 108 snapshots). The ratio is
`RMS(legoESM − NEMO) / RMS(NEMO about its own area-weighted basin mean)`; the
preregistered bar is 1e-2.

| field | RMS difference | field's own spatial scale | ratio | bar 1e-2 |
|---|---|---|---|---|
| SST | 1.008521e-03 K | 3.428892e+00 K | **2.94e-04** | met, 34× under |
| SSS | 9.717065e-05 g/kg | 3.660823e-01 g/kg | **2.65e-04** | met, 38× under |
| SSH | 1.820434e-05 m | 2.131009e-01 m | **8.54e-05** | met, 117× under |
| zonal-mean T | 5.937230e-05 K | 4.063308e+00 K | **1.46e-05** | met, 684× under |
| zonal-mean S | 7.992802e-06 g/kg | 3.843653e-01 g/kg | **2.08e-05** | met, 481× under |

Read as the paper reads it: the JAX-minus-Fortran difference sits between 3.5
and 4.8 orders of magnitude below the field's own structure, where the standard
asks for two. The two models' spatial scales agree with each other to five
figures as well (legoESM's SST scale is 3.429041 against NEMO's 3.428892), so
the denominator is not doing the work.

## 2. P2 — bounded difference. **NOT MET**.

Preregistered: over months 13-120 the relative trend of the monthly 3-D
temperature RMS is below 1 %/month in magnitude and its max/median is below 10.

| statistic | preregistered window, months 13-120 | bar |
|---|---|---|
| relative trend | **-2.81 %/month** | ±1 %/month — **not met** |
| max / median | **20.99** | 10 — **not met** |
| mean / median / max | 1.678e-03 / 6.464e-04 / 1.357e-02 K | — |

Both criteria fail. Neither failure is a sign that the difference is running
away: the trend is negative, and the record ends at its quietest. But the
preregistered gate is what it is, and it failed, so P2 is recorded as NOT MET
and the word "bounded" is not re-awarded on other evidence below.

What the series actually does, because the two failing statistics do not say
it. Annual means of the monthly 3-D temperature RMS, one per year:

```
2.74e-04  6.44e-03  3.75e-03  9.13e-04  8.92e-04
1.00e-03  6.55e-04  5.71e-04  5.07e-04  3.79e-04   K
```

The shape is **non-monotonic and dominated by a single peak**. Months 13-32
run 8.78e-03, 2.33e-03, 6.99e-03, 1.04e-02, 5.64e-03, 1.31e-03, 2.40e-03,
9.96e-03, 1.04e-02, 8.86e-03, 6.94e-03, 3.34e-03, 9.17e-04, 3.83e-04,
3.37e-04, 2.25e-04, 2.25e-04, 2.27e-04, 8.77e-03, 1.357e-02 K: it oscillates
across nearly two decades through years 2 and 3, falls as low as 2.25e-04 K at
months 28-30 — below where year 2 started — and then spikes to the record
maximum at month 32. After that it decays. Month 120 is 4.16e-04 K, 6.4 times
smaller than month 12 and 32.6 times smaller than the peak. An earlier draft of
this receipt described months 11-14 as the difference "continuing to grow";
that was wrong, and the four numbers it quoted (month 14 is below month 13)
already contradicted it.

**POST-HOC, no preregistered standing.** The same two statistics for every
possible start year, so the choice of window is not a choice:

| window | relative trend | max / median |
|---|---|---|
| years 2-10 (preregistered) | -2.81 %/month | 20.99 |
| years 3-10 | -2.34 %/month | 22.21 |
| years 4-10 | -1.20 %/month | 4.63 |
| years 5-10 | -1.43 %/month | 3.85 |
| years 6-10 | -1.74 %/month | 3.95 |
| years 7-10 | -1.33 %/month | 2.89 |
| years 8-10 | -1.63 %/month | 3.38 |
| years 9-10 | -2.77 %/month | 3.81 |
| year 10 | -0.29 %/month | 1.48 |

The spike criterion is satisfied from year 4 onward; the trend criterion is
satisfied by no window, always on the negative side. These rows were computed
after seeing the data and settle nothing — they are here so that the reader
does not have to take "the peak was early" on trust.

**Scale check, which the preregistration asks for beside the difference.** Over
months 13-120 the median difference is 6.46e-04 K while each model's own
month-to-month change has a median of 0.3133 K. The two models differ by
**0.21 % of their own monthly variability**. The volume-weighted difference,
which does not over-weight the thin surface layers, is smaller still: median
3.36e-04 K.

## 3. P3 — drift. **MET**, by three orders of magnitude.

Both models cool by the same amount over the decade: the volume-mean
temperature falls from 6.694676 °C to 6.406681 °C in legoESM and to 6.406680 °C
in NEMO, a cooling of −0.287995 K and −0.287996 K respectively.

| | measured | bar |
|---|---|---|
| max abs difference of volume-mean T, all 120 months | **1.31122e-05 K** | 1e-02 K — met |
| the same over year 1 only | **1.43305e-07 K** | 3e-03 K — met |
| max abs difference of volume-mean S, all months | 2.51262e-10 g/kg | — |

The drift curves are visually one line; their separation is 4.6e-05 of the
drift itself.

## 4. Energetics

Volume-weighted domain means over years 2-10:

| | legoESM | NEMO | relative difference |
|---|---|---|---|
| kinetic energy | 2.20108e-04 m²/s² | 2.20090e-04 m²/s² | 8.44e-05 |
| variance about the climatological flow | 1.22735e-05 m²/s² | 1.22712e-05 m²/s² | 1.89e-04 |

The second row is what the round brief called EKE. It is not an eddy kinetic
energy here: this card has ~106 km cells over a flat bottom and is sampled
monthly, so it resolves no mesoscale, and what the number measures is the
seasonal cycle plus the drift. It is reported under that name in the JSON. Both
rows use each model's own C-grid face velocities squared at coincident indices
and weighted by T-cell volume — identical on both sides, so the comparison is
fair, and not a physical energy budget, so neither number should be quoted as
one.

## 5. Seasonal cycle, years 2-10

Basin-mean SST by calendar month runs 17.327 °C (February) to 21.003 °C
(August), an annual range of 3.676 K in both models, and the two models' SST
cycles differ by at most **1.35e-04 K** — 3.7e-05 of the range. Basin-mean
mixed-layer depth runs 30.6 m (June) to 281.8 m (February), a range of 251.20 m
in both models, and the two cycles differ by at most **2.26e-02 m**, i.e.
9.0e-05 of the range. The mixed layer is the 0.2 K temperature-threshold depth,
the same definition on both sides.

## 6. The year-1 final-month jump: answered, and a weaker signal found

The question was whether the 25-fold jump at the last month of year 1 repeats
every twelfth month — which would make it an artefact of a write path or of the
calendar — or belongs only to that one snapshot.

**The 46-fold event does not recur.** Per year, the ratio of that year's
twelfth month to that year's median:

```
year   1     2     3     4     5     6     7     8     9    10
m12 / 46.01  0.48  1.23  1.03  1.74  0.95  1.98  1.90  1.29  1.16
```

Year 1 is 46×; no other year exceeds 2×. Nor was month 12 a terminal jump: the
difference goes on to reach 1.04e-02 K in year 2 and peaks at month 32. Month
12 looked like the end of a rise only because the year-1 record stopped there.

**But a weak month-12 preference does exist, and the first test was too coarse
to see it.** Normalising each month by its own year's median and taking the
median over years 2-10 gives, by calendar month:

```
month   1     2     3     4     5     6     7     8     9    10    11    12
norm  1.052 1.007 1.072 0.949 0.878 0.927 0.980 1.011 0.981 1.066 1.080 1.229
```

Month 12 is the largest of the twelve, at 1.23, and it is above its year's
median in 7 of 9 years. The effect is a 23 % modulation, not a 46-fold jump, so
it does not explain year 1 — but it is real and it is phase-locked to the
calendar. The pattern across the year is the winter half high (months 10-3,
1.01 to 1.23) and the summer half low (months 4-7, 0.88 to 0.98), which tracks
the seasonal mixed layer measured in section 5: 184-282 m in December-March
against 30-34 m in June-August. **PLAUSIBLE**, not confirmed: a deeper mixed
layer engages more of the water column in the vertical mixing operators, so a
seasonal modulation of the difference is what one would expect, but nothing
here isolates that mechanism, and the test above was designed after seeing the
series.

## 7. Is the climatology flattering itself? The instantaneous check

The P1 ratio takes an RMS **after** averaging 108 snapshots, so monthly
differences of opposite sign would cancel in the mean and leave a fine ratio
sitting on top of poor instantaneous agreement. The month-by-month surface
differences, over the same months 13-120, say how much of that happened:

| field | monthly instantaneous RMS, mean / median / max | climatological RMS | climatological / monthly mean | monthly median over the field's spatial scale |
|---|---|---|---|---|
| SST | 2.719e-03 / 6.588e-04 / 3.156e-02 K | 1.009e-03 K | 0.371 | **1.92e-04** |
| SSS | 3.609e-04 / 1.518e-04 / 4.406e-03 g/kg | 9.717e-05 g/kg | 0.269 | **4.15e-04** |
| SSH | 3.584e-05 / 3.481e-05 / 1.155e-04 m | 1.820e-05 m | 0.508 | **1.63e-04** |

Averaging does shrink the difference, by a factor of two to four against the
mean of the monthly values — that much cancellation is real and is disclosed
here rather than left for a reader to suspect. It does not carry the result:
the **median instantaneous month** already sits 5.2e+03 (SST), 2.4e+03 (SSS)
and 6.1e+03 (SSH) times below each field's own spatial scale, so the 1e-2 bar
is met snapshot by snapshot and not only after averaging.

The single worst months, which all fall in the years-2-3 peak, are the honest
exception and are reported rather than folded into a median. Measured against
each field's own spatial scale, the worst month is 9.20e-03 for SST (month 32,
the difference peak) and 5.42e-04 for SSH (month 21) — inside the 1e-2 bar —
and **1.20e-02 for SSS (month 33), outside it**. So the bar is met by the climatology, met by the
typical month, and missed by the surface salinity of the three worst months
around the difference peak.

## 8. Figures

Written to `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/decade/`, all well
under the 2 MB cap:

| file | bytes | what it shows |
|---|---|---|
| `fig1_decade_climatology_maps.png` | 119127 | years 2-10 SST, SSS, SSH: legoESM, NEMO, difference |
| `fig2_decade_difference_series.png` | 94496 | the 120-month T and S difference against each model's own month-to-month change, log scale |
| `fig3_decade_drift_energetics.png` | 83299 | volume-mean T and S and domain-mean KE, both models |
| `fig4_decade_seasonal_section.png` | 86754 | seasonal SST and mixed-layer depth, and the zonal-mean temperature difference section |

## 9. What this does not establish

- **The denominator is a surrogate.** The bar divides by the field's own
  spatial variability, not by a model-minus-observation bias as the cited
  paper does, because this case has no observations. A field with strong
  basin-scale structure gets a large denominator; both denominators are
  printed in section 1 so a reader can substitute their own.
- **The regime does not amplify divergence, and that is most of why this looks
  good.** GYRE here is a smooth, strongly and analytically forced flow on
  ~106 km cells over a flat bottom, resolving no mesoscale. Two deterministic
  models started from an identical rest state under identical forcing will
  track closely in such a regime even when their numerics genuinely differ.
  Nothing in this receipt shows the agreement would survive an eddying or
  chaotic configuration; that is the next case, not this one.
- **The preregistered P2 window was badly chosen**, and that is recorded rather
  than repaired: years 2-10 still contain the spin-up peak. A future run of this
  protocol should preregister years 5-10, or state the peak month in advance.
- The velocity rows compare C-grid face values at coincident indices. Fair
  between the models, not physical energies.
- The zonal-mean sections are weighted by layer thickness, not by row area.
  On this card the two are the same thing — the cell spacing is exactly
  106000 m in both directions at every point, measured — and the scorer now
  refuses a grid where that stops being true rather than inheriting the
  formula. On a stretched grid the narrow high-latitude rows would be
  over-weighted and the section ratios would move.
- The month-pairing test that proves the scorer is not comparing legoESM month
  *m* against NEMO month *m*±1 is skipped where the year-1 archive is absent, so
  on a machine without it that guard is unexercised.
- The acquisition's coverage in the unit test is a string tripwire over its
  refusal lines, not an execution of those refusals.
- One legoESM member, one NEMO run. This is a two-model comparison, not an
  ensemble, so nothing here separates the residual difference from what either
  model would show against a perturbed copy of itself over ten years.

## 10. Reviews

Two reviews ran on the harness **before** the decade data existed, and their
findings are closed in commits `e7023b7f5` and `044dce3d3`: sixteen items,
including two synthetic violations that could not fail for the reason they
named, an unweighted section ratio on a grid whose layers span 10 m to 300 m, a
mixed-layer fallback that understated a fully mixed column by half the bottom
layer, a binary-symbol refusal defeated by SIGPIPE under `pipefail`, and two
completeness loops that printed a count they never took. Three further points
were accepted rather than fixed and are listed in section 9.

Two further reviews audited this receipt and the numbers in it **after** the
decade was scored. Both independently recomputed every headline figure from the
JSON — the five climatology ratios, both drift bars, the window statistics, all
ten annual means, the per-year month-12 table, the post-hoc windows, the scale
check and the seasonal ranges — and both found them correct to the printed
precision, with the preregistered bar text quoted rather than restated. Their
findings changed this document in four places, and each change is visible above:
the claim that the difference "keeps growing past month 12" was false and its own
cited numbers said so; the post-hoc windows are now the full sweep of every
start year rather than two chosen ones; the calendar test was too coarse and a
real 23 % month-12 modulation was found once it was sharpened; and section 9
now names the surrogate denominator, the non-amplifying regime and the
averaging-cancellation risk, with section 7 measuring the last of those instead
of leaving it as a caveat.
