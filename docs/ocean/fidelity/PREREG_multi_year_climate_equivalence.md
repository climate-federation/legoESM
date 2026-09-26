# Preregistration: 20-year DINO climate equivalence

Date: 2026-08-30. Session:
`01a04e34-d1fb-73e0-b25a-177641f0a246`. **FROZEN BEFORE ANY CAPSTONE
ARM IS RUN.** Producer:
`ddd3a8476afd87da4afa5747eb7e5057490b893c`.

## Question, horizon, and scope

At the fully faithful defaults, is the `nemo_dino_kamm_mlf` twin bridge
statistically indistinguishable from NEMO over 20 model years when judged on
climate statistics and a noise floor measured by fresh ensembles at that same
horizon?

The horizon remains **20 years**, where one model year is NEMO's configured
360-day year (`nn_leapy=30`), hence 7,200 days and 230,400 steps after the
matched day-180 state. This is the campaign's recorded verdict tier and the
longest horizon for which the original NEMO reference has a complete matched
record. It is also well beyond the previously measured 3--5-year saturation
of a 10-year perturbation ensemble. It is not amended to 40 years: the
historical y21--40 archive starts from the from-rest trajectory, not this
matched day-180 state, and NEMO was still ramping at y40. This verdict is
therefore a **20-year-horizon climate-distribution claim**, not an equilibrium
claim.

In scope: fp64 `nemo_dino_kamm_mlf`, NEMO day-180 restart, bridged BEFORE
state, bridged TKE, corrected T-point stress reconstruction, NEMO ladder
`both`, and every current faithful card default. State-map differences may be
reported as diagnostics but cannot decide the verdict.

Out of scope: standalone `nemo_dino_kamm`, a from-rest run, production recipes
without the NEMO bridge, and cross-recipe transfer. `TRANSFER-20Y` is entered
as a separate follow-up: repeat the same families with that recipe's own
start-state ensemble and newly measured floors; no result here transfers to
it.

## Ensemble and archive admission

Each model has six members: an unperturbed control and now-level-temperature
relative kicks of `1e-14` at seeds 1, 2, 3, 4, and 5. The perturbation is the
already reviewed `perturb_nemo_tn_90d.py` / `kamm_twin_90d.py` convention.
All six legoESM members are fresh runs at the pinned producer. The common seed
labels align the perturbation construction but do not make the two models'
chaotic realizations a paired statistical sample.

NEMO members 0--2 may reuse only their `RUN_VERDICT360_M{0,1,2}` year-one
prefix. Admission requires the certified binary, source restart, namelists,
mesh, terminal `kt=17280` tile count, the three registered terminal manifest
hashes, and `STOP 0`. They then run a fresh 19-year continuation. Members 3--5
are fresh 20-year runs from the day-180 restart. Any failed prefix gate changes
that member to a fresh 20-year run; it never relaxes a hash.

The historical `RUN_20Y_REBUILD` and `RUN_40Y_REBUILD` annual means are
admitted only as calendar, throughput, and reducer plants. They began on the
from-rest trajectory and cannot be a member, baseline, or floor. Their known
y20/y40 ACC values must be reproduced by the completed scorer as a positive
control, but cannot enter a denominator or verdict.

The fresh epoch baseline is the six-member ensemble mean on each side, not an
old single control trajectory. Every legoESM artifact must stamp the pinned
producer, fp64 compute, recipe, seasonal clock, bridge start, stress carry,
TKE bridge, ladder, full resolved configuration, and the absence of CLI
physics-selector overrides. Across legoESM members, resolved configuration may
differ only in perturbation seed. Across NEMO members, namelists may differ
only in restart/step/output-control rows and perturbation receipt.

## Registered sampling

Full fp64 T/S/SSH/U/V snapshots are stored on both sides at:

- annual endpoints y1--y15: days `360,720,...,5400`;
- monthly endpoints throughout y16--y20: days `5430,5460,...,7200`.

This gives 75 matched snapshots. The registered full-state years are y1, y5,
y10, y15, and y20. The final five-year climate window is the 60 monthly
snapshots at days 5430--7200. legoESM's #1688 fp64 live-state reductions are
always on at all 75 snapshots, including the full 197-row transport vector;
`--daily-acc` is retained as a higher-cadence control. NEMO uses semiannual
restart output through y15 so the day-180 start phase still yields each annual
endpoint, then monthly output through y20.

The scorer must use the same sample dates on both sides. Missing dates,
float32 3-D storage, halo/mask disagreement, or an inferred date is fatal.

## Frozen statistic families

All geometry, masks, and reductions are imported from committed campaign
instruments. No scorer may retype them.

1. **ACC series.** The additive mean-reduced full-section transport and the
   channel-band transport from `acc_driver_decomp` / `floor90_ensemble`:
   final-five-year mean, y11--y20 least-squares annual-endpoint slope,
   12-month climatology, seasonal amplitude, deseasonalized monthly standard
   deviation, and interannual standard deviation of five annual means. The
   historical median-reduced full-section number is reported only; it is not
   additive and has no family verdict.
2. **Basin and row transports.** The exact south/channel/north partition and
   every one of the 197 `basin_seasonal_decomp.row_transport` rows: final-window
   mean and y11--y20 slope. The three basin sums must close to the full-section
   mean to `1e-9 Sv` at every snapshot. Basin seasonal amplitude,
   deseasonalized monthly standard deviation, and interannual standard
   deviation are also scored.
3. **MLD seasonal cycle.** The reviewed `mld_climate_audit.py` density-threshold
   reconstruction (`delta_sigma=0.01 kg m-3`, 10-m reference convention),
   area-weighted separately over the same south/channel/north row groups. The
   twelve final-window calendar-month means and each group's seasonal
   amplitude, deseasonalized monthly standard deviation, and interannual
   standard deviation are scored. Peak month is descriptive because it is a
   discrete circular statistic.
4. **T/S water-mass census.** Fixed reference volume
   `e1t*e2t*e3t_0*tmask`, the three row groups, and the existing
   `ts_divergence_atlas.DEPTH_CLASSES`: upper `<200 m`, interior
   `200--1400 m`, abyss `>=1400 m`. For each of the nine basin/depth cells and
   each tracer, score the final-window volume-weighted mean and deterministic
   weighted q10/q50/q90. Also score T--S covariance and cumulative volume
   fractions above the already registered sigma0 thresholds 1.20, 1.40, 1.50,
   and 1.60 kg m-3. Quantiles use the first sorted value whose cumulative
   reference volume is at least q times total wet reference volume; no
   interpolation.
5. **Upper/deep density contrasts.** The imported acceptance-gate `up` and
   `deep` reductions: final-window mean, y11--y20 annual-endpoint slope,
   12-month climatology, seasonal amplitude, deseasonalized monthly standard
   deviation, and interannual standard deviation.
6. **Variability.** The seasonal amplitude, deseasonalized monthly standard
   deviation, and five-annual-mean standard deviation named above for ACC,
   basin transports, group-mean MLD, density contrasts, and basin/depth T/S
   means. These are member-level climate statistics with their own ensemble
   floors; a mean-state pass cannot substitute for a variability pass.

For a monthly series, the climatology is the mean of the five values for each
calendar month. Deseasonalized standard deviation is sample standard deviation
over all 60 residuals. Annual means are the twelve months within each of y16
through y20. Slopes use ordinary least squares on the stated annual endpoints,
with model year as the coordinate. No time weighting or smoothing is added.

## Horizon-matched floor and frozen bands

For each scalar statistic `s`, first compute one value per member over exactly
the registered window. Let

```
gap_s   = mean(lego_s) - mean(nemo_s)
L_s,N_s = sample standard deviations across the six members
floor_s = sqrt(L_s**2 + N_s**2)
R_s     = abs(gap_s) / floor_s
```

This is the spread of a difference between one draw from each 20-year climate
distribution. No 90-day, one-year, 10-year, or from-rest floor may enter this
calculation. The scorer reports each side separately; a >10x side imbalance is
flagged `ONE_SIDED` but is not silently replaced.

Uncertainty is frozen as 20,000 deterministic nonparametric bootstrap draws,
seed 1455, independently resampling six members with replacement on each
side and recomputing gap, floor, and `R`. Same-seed members are not resampled
as pairs. Let `R_lo/R_hi` be the 2.5/97.5 percentiles:

- **CONFIRM** when `R_hi <= 2.0`;
- **REFUTE** when `R_lo > 2.0`;
- **UNRESOLVED** otherwise.

A statistic is mechanically **UNRESOLVED_QUANTIZED** if either side has fewer
than four distinct member values, if its measured floor is below ten times the
plant-measured numerical/storage quantum, if more than 1% of bootstrap draws
have zero/nonfinite floor, or if any registered input/control is missing. A
point estimate alone never promotes it.

Within each family, the simultaneous confirmation control is the 97.5th
percentile of the bootstrap maximum `R` over that family's registered scalars.
A family is CONFIRM only when that value is at most 2.0; it is REFUTE when any
member scalar is REFUTE; otherwise it is UNRESOLVED. The capstone verdict is:

- **STATISTICALLY_INDISTINGUISHABLE_AT_20Y** only if all six families CONFIRM;
- **DISTINGUISHABLE_AT_20Y** if any family REFUTES;
- **UNRESOLVED_AT_20Y** otherwise.

The family rule controls the many row/census comparisons and prevents a list
of favorable scalar passes from masking one resolved regional failure.

Endpoint floors at y1, y5, y10, y15, and y20 and their growth are mandatory
diagnostics. They never replace the final-window 20-year floor. A growing y20
floor qualifies the result as horizon-specific; it does not convert it into an
equilibrium claim.

## Fail-closed controls

Before any science verdict, the instrument must pass:

1. exact producer/config/start/clock/ladder/stress/TKE/fp64 and snapshot-date
   gates, plus NEMO binary/restart/mesh/prefix manifests;
2. same wet mask and one-ring halo convention on both loaders;
3. all finite/stable receipts and NEMO `STOP 0`;
4. basin-to-full transport closure at every snapshot;
5. reconstruction of the archived from-rest y20/y40 ACC controls without
   admitting them as arms;
6. plants that add transport to one wet U row, perturb one wet T/S census
   cell, deepen one wet-column MLD, rotate a monthly cycle, and collapse one
   ensemble to identical values. Each owned family must move and the last
   plant must reach `UNRESOLVED_QUANTIZED`;
7. classifier plants reaching CONFIRM, REFUTE, and UNRESOLVED, including the
   family maximum rule.

Any failure yields `INVALID_CAPSTONE`; no partial climate claim may be quoted.

## Cost and stopping rule

Measured one-year rates imply approximately 3.5--4.5 hours per fresh 20-year
legoESM member before snapshot overhead and 2.5--3 hours per 20-year NEMO
member on 16 ranks. Budget **12--15 hours wall** for six legoESM members in
three two-GPU waves, and **15--18 hours wall** for serial NEMO (members 0--2
reuse one year). NEMO is serial because the registered four-way run was about
24x worse in aggregate under dump-I/O contention. NEMO raw restart storage is
about 120 GiB; the handoff requires 160 GiB free and never copies an oracle
archive wholesale.

No arm is interpreted until all twelve members, the bracket, the plants, and
the frozen scorer complete. A failed or unstable member is rerun from its
registered start; it is not dropped or replaced post hoc.
