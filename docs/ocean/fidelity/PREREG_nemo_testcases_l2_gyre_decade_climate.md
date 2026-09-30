# Preregistration — GYRE decade climate-tier comparison (legoESM vs NEMO 5.0.2)

Written before either ten-year run existed. Every constant, every scored row
and every expectation below is fixed here; the scorer
(`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_decade_climate.py`)
reads them and prints numbers, and it prints **no verdict**. The receipt
interprets.

## 0. Why a second tier at all

The campaign's first tier is bit-exactness: legoESM reproduces NEMO 5.0.2 step
for step on the certified GYRE card, and a 360-day from-rest run is scored at
day 30 / 240 / 360 with a 3-D temperature RMS of
2.327677e-06 / 6.586172e-05 / 2.670992e-03 K (round 183 receipt,
`docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_round183_carried_n2_landing_receipt.md`).

That tier stops being informative once the two trajectories separate at the
rounding floor and the flow amplifies the separation. The FESOM2-JAX paper
(arXiv:2608.01546, §4) hit the same wall and answered it with a second tier:
"the two runs are not bit-identical, so we compare statistics" — climatological
mean state, multi-decadal drift, seasonal cycle — with the requirement that the
**JAX-minus-Fortran RMS sits at least two orders of magnitude below the
model-minus-observation bias**. This preregistration is that second tier for
GYRE.

GYRE is an idealised double-gyre box with no observations, so there is no
model-minus-observation bias to divide by. The substitute, fixed here and not
chosen at scoring time, is the field's **own spatial variability**: the RMS of
each model's climatological field about its own area-weighted basin mean. It is
the same quantity a model-minus-observation bias is measured against — how much
structure the field actually has — and it is computed identically on both
models.

## 1. The two runs

| | legoESM | NEMO 5.0.2 |
|---|---|---|
| card | certified GYRE card, `CASE = "GYRE-zco"`, from rest | `GYRE_OMIP_L2_P3_SM_R41ADVSP`, from rest |
| binary / tree | lane tip, fp64 + libm transcendentals, `JAX_ENABLE_X64=1`, `JAX_PLATFORMS=cpu` | `BLD/bin/nemo.exe`, sha256 `a759e8b478e3bda5ba731009fd353159db892c5ac2ba4c48351f5136411960cd` |
| timestep | 14400 s | 14400 s (`&namdom rn_Dt` in the card's `namelist_cfg`) |
| calendar | 360-day years (`nn_leapy = 30`) | same |
| steps per day | 6 (86400 / 14400) | same |
| length | 3600 days = 21600 steps | `nn_itend = 21600` |
| output cadence | snapshot every 180 steps (30 days), `day{NNN}.npz` | restart every 180 steps (`nn_stock = 180`) |
| snapshots | 120 months | 120 months |

**Deviation from the round-brief, recorded here rather than argued later.** The
brief said 14400 steps and a 120-step restart cadence on an assumed 4 steps per
day. The card's own namelist says `rn_Dt = 14400.`, so a day is **6** steps, a
decade is **21600** steps and a month is **180** steps. The namelist is the
authority; the brief's three numbers are not used.

**Snapshots only, no monthly means.** NEMO's side of this comparison is a
restart file, which is an instantaneous state. A monthly-mean legoESM field
compared against an instantaneous NEMO field would be a different quantity on
each side — the exact confound this campaign refuses elsewhere. Both models are
therefore read as instantaneous month-end snapshots, and every statistic below
is built from those.

## 2. The scored rows

Months are numbered 1..120; month *m* is model day 30·*m*. "Years 2-10" means
months 13..120, i.e. the first year (the from-rest spin-up transient) is
excluded from every climatology.

1. **Difference time series.** For each month, `RMS(legoESM − NEMO)` over all
   wet 3-D cells, for T [K] and S [g/kg]. Beside it, for scale, each model's
   own month-to-month change `RMS(x_m − x_{m−1})` in the same units. A
   difference that is small against a model's own monthly variability is small
   in the only sense that matters.
2. **Climatology, years 2-10.** Time-mean SST, SSS and SSH maps for each model;
   the difference map; `RMS(difference)`; and for each model
   `RMS(field − area-weighted basin mean)`. The reported bar quantity is
   **`ratio = RMS(legoESM − NEMO) / RMS(NEMO − basin mean of NEMO)`**. The same
   three numbers for the zonal-mean (mean over the box's x extent, wet cells
   only) temperature and salinity sections.
3. **Drift.** Volume-weighted basin-mean T and S per month, both models, and
   their difference.
4. **Energetics.** Volume-weighted domain-mean kinetic energy
   `0.5·(u² + v²)` per month, both models; and the same quantity formed from
   `u − u_clim`, `v − v_clim` where the climatology is the years-2-10 mean.
   The round brief called that second number EKE. **It is not an eddy kinetic
   energy on this card** — GYRE at this resolution has ~106 km cells over a
   flat bottom and resolves no mesoscale, and a monthly snapshot cadence
   cannot see one either — so it is reported under the name of what it
   actually measures, `velocity_variance_about_climatology`, the variance of
   the monthly flow about the record mean, i.e. the seasonal cycle plus the
   drift. Both models' velocities are read at their own native C-grid points
   and are *not* interpolated to T points; the formula is identical on both
   sides, so the comparison is fair, but neither number is a certified energy
   budget.
5. **Seasonal cycle.** Area-weighted basin-mean SST and mixed-layer depth by
   calendar month (1..12), averaged over years 2-10, both models. Mixed-layer
   depth is the depth at which temperature first falls 0.2 K below the surface
   cell, linearly interpolated between cell centres — a temperature-threshold
   definition, computed identically on both models, chosen because the npz
   snapshot and the NEMO restart both carry T and neither carries a density
   diagnostic.

## 3. Geometry and provenance, fixed here

* wet mask, `tmask` from
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_kt1_10/mesh_mask.nc`;
  `dz`, `dy`, `area`, `gdept_0` from the legoESM recipe card. These come from
  `_geometry` in `nemo_testcase_l2_gyre_year_fromrest.py` and are not
  re-derived.
* Physical constants come from `legoesm.constants`; none is written as a
  literal.
* Every JSON row carries the file it was read from, and the report carries the
  producing commit.

## 4. The bar

The FESOM2-JAX standard, restated for GYRE. Reported for SST, SSS, SSH and for
the zonal-mean T and S sections:

> **P1 — two orders of magnitude.** `ratio` ≤ 1e-2 on every climatological
> field listed in row 2.

> **P2 — bounded difference.** Over months 13..120 the monthly 3-D T RMS does
> not trend upward: the least-squares slope of that series, divided by its own
> mean, is reported as a relative trend per month, and the expectation is
> |relative trend| < 1 %/month together with
> `max(months 13..120) / median(months 13..120)` < 10. Year 1 is excluded
> because from-rest spin-up grew the difference by three decades
> (2.3e-06 → 2.7e-03 K) and extrapolating a spin-up transient into a decade
> would be a confound, not a result.

> **P3 — drift.** The two models' volume-mean temperatures stay within
> 1e-2 K of each other at every month, and within 3e-3 K over year 1 — the
> latter because a volume mean of a difference cannot exceed that difference's
> RMS, which round 183 measured at 2.670992e-03 K on day 360. Both are
> reported: `max_abs_T_volmean_difference` over all months and
> `max_abs_T_volmean_difference_year1` over months 1-12.

A number outside a bar is reported as a number, in the receipt, with the bar
quoted next to it. The tool never prints PASS, FAIL or a verdict.

## 5. Validation before the data exists

The scorer is exercised on **year 1**, which is already on disk, before either
decade run finishes: legoESM's 361 daily snapshots from round 183
(`round183/after_year/lego_seed0_year`) subset to months 1..12, against NEMO's
daily restarts from round 132 (`round132/oracle_daily_restarts`) at steps
180, 360, ..., 2160. The month-12 3-D T RMS this produces must equal the round-183
receipt's day-360 number, 2.670992e-03 K, to the printed precision. That is the
scorer's self-validation and it is quoted in the receipt.

## 6. Admission of the NEMO decade record

Per operator's note AS, a NEMO record is admitted when its restarts are
byte-identical to the un-instrumented round-132 reference at the shared steps.
The acquisition
(`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_decade_climate/run.sh`)
checks all twelve year-1 months (steps 180..2160) against
`round132/oracle_daily_restarts` and refuses on the first difference. It runs
the **already-certified binary** — the same `nemo.exe` the year-from-rest
pristine control ran, whose day-30 restart this preregistration's author
verified is byte-identical to round 132's — and builds nothing, so there is no
new compilation that could differ in the last bit.
