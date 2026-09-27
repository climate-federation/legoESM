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

## 1. P1 — the FESOM2-JAX bar: two orders of magnitude. **MET**, with room.

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

## 2. P2 — bounded difference. **NOT MET as written**, and the reason matters.

Preregistered: over months 13-120 the relative trend of the monthly 3-D
temperature RMS is below 1 %/month in magnitude and its max/median is below 10.

| statistic | preregistered window, months 13-120 | bar |
|---|---|---|
| relative trend | **−2.81 %/month** | ±1 %/month — **not met** |
| max / median | **20.99** | 10 — **not met** |
| mean / median / max | 1.678e-03 / 6.464e-04 / 1.357e-02 K | — |

Both failures are on the shrinking side, and both come from the same thing: the
window I preregistered as "after spin-up" is not after spin-up. The difference
grows through year 1, peaks at **month 32** at 1.357e-02 K, and then decays for
the rest of the decade. Annual means of the monthly 3-D temperature RMS, one
value per year:

```
2.74e-04  6.44e-03  3.75e-03  9.13e-04  8.92e-04
1.00e-03  6.55e-04  5.71e-04  5.07e-04  3.79e-04   K
```

Month 120 is 4.16e-04 K, which is **six times smaller than month 12** (2.67e-03
K) and 33 times smaller than the peak. The difference is bounded; the
preregistered instrument reports a large negative trend and a large spike ratio
because its window still contains the rise and the peak.

**POST-HOC, not preregistered, labelled as such.** The same two statistics on
later windows, reported because the reader will otherwise have to compute them:

| window | relative trend | max / median |
|---|---|---|
| years 5-10 (months 49-120) | −1.43 %/month | 3.85 |
| years 6-10 (months 61-120) | −1.74 %/month | 3.95 |

The spike ratio falls inside the bar once the peak is excluded; the trend stays
outside it, still negative, because the series is genuinely decaying. These are
post-hoc windows chosen after seeing the data and carry no preregistered
standing. What stands preregistered is the row above: not met, on the shrinking
side.

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

## 6. The year-1 final-month jump: answered

The question was whether the 25-fold jump at the last month of year 1 repeats
every twelfth month — which would make it an artefact of a write path or of the
calendar — or belongs only to that one snapshot.

**It is neither a repeating calendar feature nor an artefact of the last
snapshot. It is the start of the growth phase.** The month-by-month values
around it are 1.055e-04 K (month 11), 2.671e-03 K (month 12), 8.782e-03 K
(month 13), 2.325e-03 K (month 14): the difference keeps growing past month 12
and peaks at month 32. Month 12 looked like a terminal jump only because the
record stopped there.

Tested directly, per year, as the ratio of that year's twelfth month to that
year's median:

```
year   1     2     3     4     5     6     7     8     9    10
m12 / 46.01  0.48  1.23  1.03  1.74  0.95  1.98  1.90  1.29  1.16
```

Year 1 is 46×; no other year exceeds 2×, and the year's largest month falls on
month 12 in only two of the remaining nine years (7 and 8), which is what
chance gives. There is no twelve-month periodicity in the difference.

## 7. Figures

Written to `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/decade/`, all well
under the 2 MB cap:

| file | bytes | what it shows |
|---|---|---|
| `fig1_decade_climatology_maps.png` | 119127 | years 2-10 SST, SSS, SSH: legoESM, NEMO, difference |
| `fig2_decade_difference_series.png` | 94496 | the 120-month T and S difference against each model's own month-to-month change, log scale |
| `fig3_decade_drift_energetics.png` | 83299 | volume-mean T and S and domain-mean KE, both models |
| `fig4_decade_seasonal_section.png` | 86754 | seasonal SST and mixed-layer depth, and the zonal-mean temperature difference section |

## 8. What this does not establish

- **The preregistered P2 window was badly chosen**, and that is recorded rather
  than repaired: years 2-10 still contain the spin-up peak. A future run of this
  protocol should preregister years 5-10, or state the peak month in advance.
- The velocity rows compare C-grid face values at coincident indices. Fair
  between the models, not physical energies.
- The zonal-mean sections are weighted by layer thickness, not by cell area;
  the scorer checks that this card's cells are equally wide before using the
  unweighted zonal mean, so the check is live, but a stretched grid would need
  the area weights added.
- The month-pairing test that proves the scorer is not comparing legoESM month
  *m* against NEMO month *m*±1 is skipped where the year-1 archive is absent, so
  on a machine without it that guard is unexercised.
- The acquisition's coverage in the unit test is a string tripwire over its
  refusal lines, not an execution of those refusals.
- One legoESM member, one NEMO run. This is a two-model comparison, not an
  ensemble, so nothing here separates the residual difference from what either
  model would show against a perturbed copy of itself over ten years.

## 9. Reviews

Both reviews ran on the harness **before** the decade data existed, and their
findings are closed in commits `e7023b7f5` and `044dce3d3`: sixteen items,
including two synthetic violations that could not fail for the reason they
named, an unweighted section ratio on a grid whose layers span 10 m to 300 m, a
mixed-layer fallback that understated a fully mixed column by half the bottom
layer, a binary-symbol refusal defeated by SIGPIPE under `pipefail`, and two
completeness loops that printed a count they never took. Three further points
were accepted rather than fixed and are listed in section 8.
