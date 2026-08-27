# Pre-registration — what the channel row-gap reorganization tracks

Committed before this lane computes any seasonal driver or model-anomaly
statistic.  The upstream channel profiles and their already-published pairwise
correlations are prior evidence, not results of this lane.

Everything is offline from the saved verdict360 states.  No model is stepped;
CPU only.  The scored horizons and members are exactly the channel-rescore
ones: days 90, 180, 270 and 360, with all four paired members on each side.

## 1. Pinned upstream machinery and fixed target

The probe must import, by committed Git object and SHA-256 rather than by a
mutable sibling checkout:

* `channel_rescore.py` at `97d3d7188eb5b9fe74a4f505a383342c81af270b`,
  SHA-256 `a3b51de6292df22b90fa9bd385dcb22ab47a3049c737f2e2abf1a28eb5a18be9`;
* `regional_audit.py` at `102ef501a60c915a01f740115124cd99d95f2197`,
  SHA-256 `2f2bbc03afed792ea0148002029a6ad6481c93bd2ee37679b4bf615aab392fe7`;
* the channel artifact at the first commit above, SHA-256
  `fd39b99842e7c20a5f42b069f8ff732e6c515dee0b0ab00bea1ddae2f52ef11b`.

The target is the same paired-member row-transport gap as the rescore:

```
G[m,t,j] = row_transport(lego[m,t], j) - row_transport(NEMO[m,t], j)
```

on imported channel rows 14--48.  The reducer retains its `e3t_1d` thickness,
`e2u` width, common wet mask and mean over longitudes 2..-2.  The probe must
reproduce every upstream `G` value before a seasonal result is read.

Seasonal evolution removes each row's four-horizon mean:

```
G'[m,t,j] = G[m,t,j] - mean_t G[m,t,j].
```

This removes the fixed row bias.  No trend is fitted: with only one annual
cycle, a linear trend and season are degenerate.  Every positive result is
therefore a phase-tracking result, not a causal or equilibrium-climate claim.

## 2. Wind is excluded before scoring

The shipped card selects `forcing_annual_cycle=True`, but its live forcing path
recomputes only `T*` and `Q_sr` from time.  `dino_wind_stress(lat_deg, cfg)` has
no time argument and is precomputed from fixed latitude knots.  The oracle
member's namelist selects `nn_forcingtype=4`; in the active `usrdef_sbc.F90`
arm, `utau` is `znl_cbc(fixed knots, gphiu)` and contains neither seasonal
cosine.

The probe verifies those source/card receipts and records their SHA-256.  A
synthetic wind callable with a time argument must make the structural guard
fail.  If the receipts hold, zonal-mean zonal wind work is **EXCLUDED A PRIORI**
as a seasonal forcing driver; it is not computed and cannot win post hoc.

The remaining scored drivers are:

1. thickness-weighted density inventory;
2. thickness-weighted vertical stratification;
3. mixed-layer depth;
4. meridional density gradient expressed as thermal-wind transport per
   adjacent-row face.

## 3. The trivial explanation is scored first

For each model and member, define its own seasonal row-transport anomaly

```
A_side[m,t,j] = U_side[m,t,j] - mean_t U_side[m,t,j].
```

The two models rotate the **same way** only if the phase score in section 5,
with legoESM `A` as target and NEMO `A` as predictor, CONFIRMS for the
four-member mean and separately for all four paired members.

Define the common rotating field and its difference:

```
C = (A_lego + A_NEMO) / 2
D = A_lego - A_NEMO = G'.
```

Fit one zero-intercept coefficient over all four horizons and rows,
`beta = sum(C*D)/sum(C*C)`.  The named proportionality residual is
`NRMSE = ||D-beta*C||/||D||`.

* **CONFIRM fixed-fraction rotating field:** same-way rotation CONFIRMS; the
  phase score from `C` to `D` CONFIRMS; and pooled plus all four member NRMSEs
  are `<= 0.30` (the imported low bar).
* **REFUTE:** either phase relation REFUTES, or pooled and at least three of
  four member NRMSEs are `>= 0.70` (the imported high bar).
* Otherwise **UNRESOLVED**.

This leg is evaluated and printed before any candidate driver.  A confirmation
means the gap is a fixed fraction of a shared rotating transport field, not an
independently rotating gap.

## 4. Driver profiles — no layer averages

Each state is reduced on the shared audit mask.  `rho` is imported from
`acc_thermal_wind.rho_of`, i.e. the campaign's card-pinned NEMO SEOS and common
pressure/depth convention.

For each driver `Q`, attribution uses its cross-model seasonal anomaly:

```
Qgap[m,t]  = Q_lego[m,t] - Q_NEMO[m,t]
Q'[m,t]    = Qgap[m,t] - mean_t Qgap[m,t].
```

The four fixed profiles are:

* **Density inventory [kg m-3].** Per row, `sum(V*rho)/sum(V)` over all wet
  channel cells, with `V=e1t*e2t*e3t_0`.
* **Vertical stratification [kg m-4].** Per row, the volume-weighted least-
  squares slope of density against positive-down `gdept_0`:
  `sum(V*(z-zbar)*(rho-rhobar))/sum(V*(z-zbar)^2)`.  This is a thickness-
  weighted column stratification measure, not a layer average.
* **Mixed-layer depth [m].** The canonical
  `legoesm.ocean.diagnostics.mixed_layer_depth`, applied identically to both
  saved states with the campaign/NEMO `mldr10_1` convention: density threshold
  0.01 kg m-3 relative to 10 m, the same SEOS, shared wet mask and actual
  column depth.  Row profiles are surface-area weighted.
* **Thermal wind [Sv].** `acc_thermal_wind.thermal_wind_rows` on each adjacent
  T-row pair 14--15 through 47--48, then the recorded longitude mean.  The
  target is placed on the same 34 faces as `(G'[j]+G'[j+1])/2`.  This reuses
  the campaign's thickness- and depth-weighted thermal-wind integrand.

Density inventory and stratification are separate verdict legs; neither may
borrow the other's score.  No best-of-family selection is allowed.

## 5. Discriminating statistic and effective sample size

For a target profile `x_t` and driver profile `q_t`, compute ordinary centered
Pearson `r_t(s)` at each of the four horizons after cyclically shifting the
driver horizon by `s=0,1,2,3`.  The exact Pearson and Bartlett adjacent-row
effective sample size are imported from `channel_rescore`; a constant profile
is UNMEASURABLE.

For each shift, combine the four correlations in Fisher-z space with weights
`max(N_eff-3, 0)`.  The phase score is the absolute back-transformed weighted
mean, so the physical relation may have either one fixed sign, but may not pick
a different sign each season.  For `s=0`, orient all four correlations by that
single pooled sign.

The cyclic null is `max(score(s=1), score(s=2), score(s=3))`.  Spatial
uncertainty is a 5,000-draw contiguous moving-block bootstrap with seed 1226,
using the same row indices in every horizon and both fields.  Block length is
`ceil(n_rows/median(N_eff))`, clipped to [2,12], as in the channel rescore.
Every retained/dropped draw and the median/minimum `N_eff` is recorded.

Four driver legs are tested, so the primary advantage interval is the
Bonferroni family-wise 98.75% interval.  The statistic is

```
Delta = score(correct phase) - max score(non-zero cyclic shift).
```

For one ensemble or member:

* **CONFIRM:** correct-phase score `>= 0.70`, every one of the four oriented
  horizon correlations is `> 0.30`, and the 98.75% block-bootstrap lower bound
  on `Delta` is `> 0`.
* **REFUTE:** correct-phase score `< 0.30`, or the 98.75% upper bound on
  `Delta` is `< 0`.
* Otherwise **UNRESOLVED**.

A driver leg **CONFIRMS** only if the four-member-mean profile and all four
paired members independently CONFIRM.  It **REFUTES** only if the four-member
mean REFUTES and at least three of four members REFUTE.  Everything else is
**UNRESOLVED**.  Thus a mean cannot hide a dissenting member and one noisy
member cannot manufacture a refutation.

## 6. Unsaturated flags and vocabulary

The upstream target carries its measured `0/35` saturated-row result, never a
hardcoded label.  For every driver and spatial position, per-side member spread
at days 180/270/360 is passed verbatim to `regional_audit.saturated`.  The
artifact reports saturated/unsaturated counts and the reason for each `u` flag.

The phase statistic does not divide by an ensemble floor, so unsaturation does
not algebraically change its correlation.  It still travels with every verdict:
`CONFIRMS_PHASE_TRACKING (u)`, `REFUTES_PHASE_TRACKING (u)`, or `UNRESOLVED
(u)`.  “CONFIRMS” means only that seasonal phase alignment beats the cyclic
null under the registered statistic.  It never means causal owner.

## 7. Controls — all run before real scores

1. **Pinned imports:** one-byte source mutation fails each SHA-256 guard.
2. **Upstream artifact:** every loaded `G[m,t,j]` reproduces the committed
   channel artifact; one planted changed value fails.
3. **Provenance/clock:** all loader stamp refusals run; the oracle restart reads
   15,552,000 s = 180 d; producer cleanliness is checked at start and end.
4. **Wind exclusion:** the real wind signature has no time argument; a planted
   time-dependent signature fails.
5. **Thickness weighting:** a surface-only synthetic density perturbation gives
   a different answer under a layer average; replacing the volume reducer with
   that layer average fails.
6. **MLD mobility:** a planted subsurface density crossing moves the canonical
   MLD by more than one metre; a uniform fully mixed column does not supply the
   same answer.
7. **Thermal-wind mobility:** a planted meridional density perturbation moves
   its adjacent face while a constant-density field gives zero.
8. **Effective rows:** a block-repeated profile has `N_eff < n_rows`; asserting
   independent rows fails.
9. **Cyclic decider:** a known smooth four-phase pattern CONFIRMS only at the
   correct alignment; relabelling its driver by one phase does not return the
   same verdict.
10. **Finite/fp64:** every state-derived profile, bootstrap accumulator and
    output array is finite fp64; NaN is fatal.

Every planted violation prints `PLANT FIRED`; a plant that does not fire aborts.

## 8. Required report

The result reports, in order: the trivial leg; wind exclusion; the four driver
legs; member replication; exact phase correlations and cyclic scores; honest
effective rows; `u` flags; thickness/area weighting; provenance stamps read and
recorded; and all deviations/retractions.  One annual cycle cannot separate
season from elapsed time, so even a confirmed driver remains a phase tracker.

## 9. Post-review amendment — 2026-08-27

This amendment is necessarily post-result and does not change a computed
correlation, cyclic score, bootstrap interval, saturation flag or weighting.
It corrects the sensitivity control and the strength of the reported labels in
response to the adversarial `SHIP AFTER CORRECTIONS` review.

The original section 7.9 registered a **smooth** known-phase plant.  The first
implementation instead used iid Gaussian rows with `N_eff=35` and exact
per-horizon `r=1`; that substitution overstated the confirm-arm sensitivity and
is retracted.  Its replacement is fixed before its values are computed:

* Six deterministic realizations use seeds 1455 through 1460 and 35 rows.
* Independent target and noise innovations are each smoothed along rows as an
  AR(1) recursion with coefficient `phi=0.85`, standardized by horizon, and
  projected onto the same four-horizon anomaly subspace by removing each
  row's horizon mean.
* For planted tracking strengths `rho = 0.60, 0.75, 0.90, 0.95, 0.975, 0.99`,
  the driver is `rho*x + sqrt(1-rho^2)*noise`, followed by the same anomaly
  projection and the production `phase_decider` with 5,000 moving-block draws,
  seed `1226 + 100*realization + round(1000*rho)`.
* The **detectable-effect floor** is the lowest planted `rho` for which all six
  realizations satisfy the original confirm arm.  If none does, the floor is
  reported as greater than 0.99.  Confirmation counts, matched-score ranges,
  effective-row ranges and all six outcomes at each strength are retained in
  the artifact.  This control diagnoses power; it cannot promote a real leg.

Because the driver advantage intervals all straddle zero and cyclic-null scores
can themselves exceed the old 0.30 low bar, the old `REFUTES_PHASE_TRACKING`
driver wording is withdrawn.  A real driver leg that entered that arm is now
reported as **NO DETECTED PHASE RELATION — indistinguishable from random
quarter relabelling**.  The registered raw classification remains in the
artifact for auditability.  The four anomaly horizons have only three degrees
of freedom after mean removal; the Fisher aggregate's four entries are
dependent, so its score is descriptive and the block interval is not evidence
for four independent seasonal observations.

The wind rationale is also narrowed: time-independent stress does not make
wind work `tau*u` time-independent because `u` evolves.  Wind is excluded only
as a prescribed **forcing-gap** driver because the same `tau` is applied on
both sides, making the imposed stress gap identically zero.  Controls and their
measured outputs must be embedded in the committed artifact.

Finally, the first clean production attempt exposed a registered no-information
case with every Fisher weight zero at `N_eff=3`.  The subsequently added
`UNRESOLVED_EFFECTIVE_N` category is explicitly accepted here as a prereg
amendment: it reports no score and cannot enter either confirmation or
non-detection counts.  No threshold was changed to create that category.
