# DINO channel seasonal attribution result

**Verdict: the channel transport gap genuinely reorganizes through the four sampled seasons, but none of the tested state-derived candidates explains its phase rotation.** The trivial fixed-fraction explanation is **REFUTED**. Prescribed wind stress is **EXCLUDED A PRIORI (time-independent)**. Density inventory, vertical stratification, mixed-layer depth, and meridional density gradient/thermal wind each **REFUTE PHASE TRACKING (u)** under the preregistered cyclic-shift test. Here `(u)` means every tested channel row was unsaturated.

This is an attribution exclusion result, not evidence that the reorganization has no physical driver. It narrows the next search to mechanisms not represented by these instantaneous row-local state gaps, such as transient momentum balances, eddy/form stress, or lagged/nonlocal adjustment.

## Target reconstructed at all horizons

The committed artifact contains every 35-row profile for all four members, both models, and all four horizons. The exact SHA-pinned upstream row reducer was used, and its stored gap profiles were reproduced with maximum absolute residual `0.0 Sv`.

| Horizon | raw gap gross, mean ± member SD (Sv) | seasonal-anomaly gross, mean ± member SD (Sv) | ensemble anomaly RMS (Sv/row) |
|---:|---:|---:|---:|
| day 90 | 0.08096 ± 0.00030 | 0.08782 ± 0.00281 | 0.004183 |
| day 180 | 0.07205 ± 0.00475 | 0.03944 ± 0.00299 | 0.002416 |
| day 270 | 0.12654 ± 0.00918 | 0.06932 ± 0.00687 | 0.003884 |
| day 360 | 0.13470 ± 0.01091 | 0.10696 ± 0.00840 | 0.005385 |

The ensemble seasonal-anomaly profile correlations for days `(90, 180, 270, 360)` are:

```text
          90       180       270       360
90     1.0000    0.0395   -0.6152   -0.2981
180    0.0395    1.0000    0.6362   -0.9165
270   -0.6152    0.6362    1.0000   -0.5449
360   -0.2981   -0.9165   -0.5449    1.0000
```

Thus the seasonal component rotates strongly even though the raw gap includes a persistent background structure. The day-360 raw gross exactly recovers the upstream `0.134695 ± 0.010907 Sv` result. Target floor saturation is `0/35` rows.

## Trivial explanation tested first

**REFUTES_FIXED_FRACTION_EXPLANATION.** For each model, its own seasonal row-transport anomaly was formed before differencing. The common rotating field was `C=(A_lego+A_nemo)/2`, and the gap anomaly was `D=A_lego-A_nemo`.

- The zero-intercept fixed-fraction fit `D=beta*C` has pooled `beta=-0.00509` and NRMSE `0.98356`; member NRMSE values are `0.98519, 0.98869, 0.98705, 0.97087`. All exceed the preregistered `0.70` refutation bar.
- Direct phase tracking of `D` by `C` is refuted for the ensemble mean and all four members. For the mean, the matched score is `0.17890`, the best cyclic-null score is `0.33858`, and matched-minus-null is `-0.15968` with Bonferroni `98.75%` moving-block interval `[-0.38195, 0.04421]`; median/minimum effective row counts are `6.49/4.10`.
- Whether legoESM and NEMO themselves rotate identically is **UNRESOLVED_EFFECTIVE_N**: their highly autocorrelated patterns reduce every Bartlett effective row count to the registered floor of 3, leaving zero Fisher weight. This does not rescue the fixed-fraction hypothesis because both its proportionality residual and the directly measurable `C -> D` phase relation fail independently.

The conclusion is therefore not “a fixed percentage of one shared rotating transport field.” The difference field has its own seasonal structure.

## Wind exclusion

**EXCLUDED_A_PRIORI_TIME_INDEPENDENT.** The legoESM DINO card has `forcing_annual_cycle=True`, but `dino_wind_stress(lat_deg, cfg)` has no time argument and the annual-cycle branch recomputes only restoring temperature and shortwave heat flux. The NEMO DINO surface-boundary source likewise assigns zonal stress from fixed latitude knots, with no seasonal time factor. Source and namelist SHA-256 receipts are recorded in the artifact. The scored candidates remaining after this exclusion were density inventory, vertical stratification, mixed-layer depth, and meridional density gradient/thermal wind.

## Driver discrimination

For each candidate, the model-to-model driver gap was converted to its four-horizon seasonal anomaly and compared row-for-row with the transport-gap anomaly. The matched phase score is the absolute effective-`n`-weighted Fisher aggregate over all four horizons. It had to reach `0.70`, have all four oriented horizon correlations above `0.30`, and beat every nonzero cyclic shift with a positive Bonferroni `98.75%` moving-block lower bound. A score below `0.30` is a registered refutation. All results below hold for the ensemble mean and all four members, each with 5,000/5,000 retained bootstrap draws.

| Leg | verdict | matched score | oriented horizon correlations (90/180/270/360) | best cyclic score | matched − null (98.75% interval) | median/min effective rows | saturation |
|---|---|---:|---|---:|---|---:|---:|
| density inventory | **REFUTES_PHASE_TRACKING (u)** | 0.0576 | -0.690 / 0.131 / -0.336 / 0.144 | 0.3333 (shift 3) | -0.2757 `[-0.4803, 0.2091]` | 9.16 / 3.39 | 0/35 |
| vertical stratification | **REFUTES_PHASE_TRACKING (u)** | 0.1227 | 0.638 / 0.004 / 0.307 / 0.157 | 0.1984 (shift 1) | -0.0757 `[-0.3553, 0.2516]` | 8.04 / 3.71 | 0/35 |
| mixed-layer depth | **REFUTES_PHASE_TRACKING (u)** | 0.0610 | 0.011 / 0.149 / 0.288 / -0.009 | 0.3492 (shift 1) | -0.2882 `[-0.5235, 0.2068]` | 24.86 / 11.21 | 0/35 |
| thermal wind | **REFUTES_PHASE_TRACKING (u)** | 0.0403 | -0.704 / 0.113 / -0.107 / 0.389 | 0.1438 (shift 3) | -0.1035 `[-0.2521, 0.2421]` | 24.07 / 4.98 | 0/34 |

No driver merely misses a confidence bound while retaining a large matched effect: every matched mean score is below the preregistered `0.30` refutation bar, the correlations change sign across horizons, and every member is also labeled `REFUTES_PHASE_TRACKING`. Thermal wind has 34 adjacent-row faces rather than 35 cell rows.

## Weighting, provenance, and controls

- Density inventory uses thickness/volume weighting in the channel band.
- Vertical stratification is the volume-weighted density-versus-depth regression, not a layer mean.
- MLD uses the canonical `delta_sigma=0.01 kg m-3` relative to 10 m, the shared seawater EOS, actual water-column thickness, and horizontal area weighting.
- Thermal wind reuses the committed meridional-density-gradient reducer and face weights.
- The exact committed audit/atlas/channel sources were loaded from git objects and SHA-256 checked: regional audit `102ef501...` / `2f2bbc03...`, atlas `102ef501...` / `72c35e61...`, channel rescore `97d3d718...` / `a3b51de6...`, upstream artifact `97d3d718...` / `fd39b998...`.
- All input provenance stamps were read and recorded, including legoESM launch `a7b940f7...`, four member paths per side, oracle restart clock `15,552,000 s = 180 d`, and the registered NEMO member-3 naming exception.
- Controls proved the machinery can fail: an aligned synthetic phase plant confirmed, a one-quarter cyclic relabeling did not retain that verdict, constant-row input returned unmeasurable, a perturbed upstream gap failed exact reproduction, and dirty/changed producer trees are fatal.

The producer was clean at commit `e9f0574af7480163fb160faa726f9a0e4cc8797c`. The first clean attempt emitted no result because a test relation had `N_eff=3` and zero Fisher weight; the committed correction reports that registered no-information case as `UNRESOLVED_EFFECTIVE_N` rather than crashing or fabricating a score. No verdict threshold changed.

Artifact: `docs/ocean/fidelity/dino_channel_seasonal_artifact.json`, SHA-256 `7eb2ec619ccf74fb4ef5613e4a3c7d50fe78a10efbd2b9908c89568518da6466`.
