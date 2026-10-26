# Pre-registration — per-row rescore of the circumpolar channel

Registered before this probe computes any channel-rescore statistic.  The
regional audit has already established the motivating facts: the channel-band
net retains about 2% of its summed per-row gap magnitude at day 360, and 17 of
35 rows exceed five times their own floor.  Those are upstream observations,
not results of this lane.  No stability, structure, roughness, or sensitivity
number defined below has been computed at registration time.

Instrument: `channel_rescore.py` (to be added after this file is committed).
Upstream instrument reused without re-derivation:
`regional_audit.py` at commit `102ef501a`.

Everything is offline from the saved verdict360 states.  No model is stepped;
CPU only.

## 1. Question and fixed data

Does the channel's small band sum represent the same row-level physical
pattern in both models, or is it a residue of row-scale noise that decorrelates
across time and perturbed members?

The channel is exactly `regional_audit.BANDS[1]`, hence the recorded
`acc_driver_decomp.LAT_GROUPS[1]`: T-rows 14--48 inclusive (35 rows).  It is
imported by reference, never retyped.  The four registered horizons are days
90, 180, 270, and 360.  The four members on each side are the verdict360
control plus three 1e-14 temperature-nudge members.

For paired member `m`, horizon `t`, and channel row `j`, define

```
G[m,t,j] = row_transport(lego[m,t], j) - row_transport(NEMO[m,t], j)
```

using `regional_audit.load_lego`, `load_nemo`, and `row_transports` verbatim.
That reducer is the verdict transport integrand: wet-mask, `e3t_1d` thickness,
and `e2u` width weighted, then mean-reduced over the same trimmed longitudes.
There is no layer average.  Pairing is by recorded member index, as in the
regional audit.  All 16 cross-side pairings are a secondary robustness check
and cannot replace the registered four pairs.

The per-row floor is also the regional audit's exact construction:

```
F[t,j] = sqrt(std_m(row_lego[m,t,j])^2
              + std_m(row_NEMO[m,t,j])^2),  ddof=1 on each side
```

No band or global floor is transferred to a row.  The band floor used in the
sensitivity statement is computed from the four members' exact channel sums,
again as the RSS of the two sides' sample standard deviations.

Every floor has `n=4` per side and is a factor-of-two estimate.  The regional
audit found every band floor unsaturated at day 360.  Therefore every
gap/floor multiple and every count beyond a floor bar is reported as an
**upper bound**.  Absolute gaps and pattern correlations are not upper bounds.

## 2. Correlation denominator and adjacent-row effective sample size

Every pattern correlation is the ordinary signed Pearson correlation after
removing each 35-row profile's own mean:

```
r(x,y) = sum_j((x_j-xbar)(y_j-ybar))
         / sqrt(sum_j(x_j-xbar)^2 * sum_j(y_j-ybar)^2)
```

The denominator is therefore the geometric mean of the two profiles' centered
sum-of-squares.  An exactly constant profile is **UNMEASURABLE**, never assigned
zero correlation.

Adjacent rows are not independent.  For every correlation, the spatial
effective sample size is

```
D = max(1, 1 + 2*sum_{k=1}^{34} (1-k/35)*rho_x(k)*rho_y(k))
N_eff = min(35, max(3, 35/D))
```

where each `rho(k)` is that centered profile's lag-k autocorrelation with its
own lagged sum-of-squares denominator.  The `max(1, ...)` rule is conservative:
negative autocorrelation may not create more than 35 independent rows.
Fisher-z intervals use `SE = 1/sqrt(N_eff-3)`; `N_eff <= 3` yields the full
interval `[-1,1]`, not an invented precision.

For the aggregate deciders below, uncertainty is a 95% spatial moving-block
bootstrap interval over 5,000 deterministic draws (seed 1226).  A draw samples
contiguous blocks, applies the same row indices to every profile, and recomputes
the complete median statistic.  Block length is
`ceil(35 / median(N_eff))`, clipped to `[2, 12]`.  This prevents the 35 rows or
the 24 overlapping correlations from being treated as independent replicates.
The probe prints the individual `N_eff`, their median/minimum, block length,
and both Fisher and block intervals.

## 3. Decider A — persistence across the four horizons

For each of four paired members, compute all six signed correlations between
its row-gap profiles at the four horizons (24 correlations total).  The primary
statistic `R_time` is their median.  Its denominator is the Pearson denominator
in section 2; its uncertainty denominator is the autocorrelation-adjusted
`N_eff`, never 35 independent rows.

* **CONFIRM persistent pattern:** `R_time >= 0.70` and the 95% block-bootstrap
  lower bound is `> 0.30`.
* **REFUTE persistent pattern / CONFIRM decorrelation:** `R_time <= 0.30` and
  the 95% upper bound is `< 0.70`.
* Otherwise **UNRESOLVED**.  A high point estimate with too few effective rows
  is not promoted.

## 4. Decider B — persistence across ensemble members

At each horizon, compute all six signed correlations among the four paired
member gap profiles (24 correlations total).  The primary statistic `R_member`
is their median, with the same correlation and effective-sample-size
denominators as section 2.

* **CONFIRM member-stable pattern:** `R_member >= 0.70` and the 95% block lower
  bound is `> 0.30`.
* **REFUTE member-stable pattern / CONFIRM member noise:** `R_member <= 0.30`
  and the 95% upper bound is `< 0.70`.
* Otherwise **UNRESOLVED**.

The all-16-cross-pairing median is reported only as a robustness check.  A
registered verdict may be weakened if it changes category, never strengthened.

## 5. Decider C — bathymetry/jet structure versus grid-scale alternation

The physical predictors are fixed before measurement:

1. row transport capacity, `A[j] = row_transports(u=1, wet)[j]`, which is the
   exact thickness-and-width-weighted wet cross-sectional area in Sv per m/s;
2. NEMO control-member per-row transport `N[t,j]`;
3. its centered row derivative `dN/dj = numpy.gradient(N[t,:])`.

At each horizon the ensemble-mean gap profile is correlated with all three.
`R_phys` is the largest absolute day-360 correlation.  Because three predictors
are tried, its interval is the Bonferroni 98.33% Fisher interval, using the
same `N_eff` denominator as section 2.  The winning predictor must also have
the same correlation sign at least three of four horizons.

Grid-scale content is

```
Z = sum_j (g[j+1]-g[j])^2 / (4*sum_j (g[j]-gbar)^2)
```

on the day-360 ensemble-mean gap.  Its denominator is four times the centered
profile energy; a pure row-to-row checkerboard approaches 1, white row noise
approaches 0.5, and a smooth profile approaches 0.  The companion sign-flip
fraction `Q_flip` counts adjacent pairs for which both rows clear `2*F`; its
denominator is the number of such eligible adjacent pairs, printed explicitly.

* **CONFIRM physical spatial structure:** (`R_phys >= 0.70`, its corrected
  lower confidence bound exceeds 0.30, and its sign is stable at 3/4 horizons)
  **or** (`Z <= 0.25`), provided `Z < 0.60`.
* **REFUTE physical structure / CONFIRM grid-scale alternation:** all three
  predictor correlations are `<= 0.30` in magnitude, their corrected upper
  bounds are `< 0.70`, `Z >= 0.60`, and `Q_flip >= 2/3` with at least six
  eligible adjacent pairs.
* Otherwise **UNRESOLVED**.  In particular, a coherent pattern not described
  by these three predictors is not called noise.

## 6. Decider D — row-level agreement and the magnitude X

At day 360, `P2` is the fraction of the 35 channel rows satisfying
`|G[control,360,j]| <= 2*F[360,j]`; its denominator is exactly 35 rows.
`P5` is the fraction exceeding `5*F`, with the same denominator.  Because the
floors are unsaturated, `P2` is a lower bound on agreement and `P5` an upper
bound on disagreement.

* **CONFIRM genuine row-level agreement:** `P2 >= 0.90` and `P5 <= 0.10`.
* **REFUTE genuine row-level agreement:** `P5 >= 0.25`.
* Otherwise **UNRESOLVED**.

The reported real-pattern magnitude `X` is the day-360 gross row disagreement

```
X = sum_j |G[control,360,j]|  [Sv].
```

Its denominator in the dimensionless cancellation statement is `X` itself.
The conservative floor-excess companion is
`X_2 = sum_j max(|G|-2F, 0)` Sv; its denominator is also `X` when reported as a
fraction.  `X` is an absolute recorded-state discrepancy; `X_2` is a lower
bound on resolvable magnitude because the floors remain unsaturated.

## 7. Decider E — sensitivity of the band sum

For each horizon and paired member, define band net `B=sum_j G[j]`, gross
disagreement `X=sum_j|G[j]|`, and cancellation survival `C=|B|/X`.  The named
denominator of `C` is the gross row disagreement, not the 34 Sv absolute
transport.

At day 360 also report:

* `J = max_j |G[j]| / F_band`, the largest one-row deletion movement, with the
  exact channel-band floor as denominator;
* `E = max |B_variant-B| / F_band` over the nine masks whose lower and upper
  edges independently move by -1, 0, or +1 row, with the original band floor
  as denominator.  This is a sensitivity diagnostic only; moved masks do not
  redefine the registered channel.

* **CONFIRM cancellation-sensitive band sum:** median `C` across the 16
  horizon/member cases is `<= 0.10`, and day-360 control has `J >= 2` or
  `E >= 2`.
* **CONFIRM robust band sum / REFUTE cancellation sensitivity:** median `C` is
  `>= 0.50`, and day-360 control has both `J <= 1` and `E <= 1`.
* Otherwise **UNRESOLVED**.

The 17 ppm full-year number is upstream and post-hoc; this lane does not
recompute it from only four horizons.  The sensitivity statement says whether
the same channel functional is robust at the registered horizons.

## 8. Final verdict rule

* **CONFIRMED persistent structure; 17 ppm masks a real pattern error of
  magnitude X** only if Deciders A and B both CONFIRM persistence, Decider C
  CONFIRMS physical structure (or is UNRESOLVED but explicitly not grid-scale),
  Decider D REFUTES genuine row-level agreement, and Decider E CONFIRMS
  cancellation sensitivity.
* **CONFIRMED decorrelating noise; genuine agreement at row level** only if A
  and B both CONFIRM decorrelation, D CONFIRMS genuine row-level agreement, and
  E REFUTES cancellation sensitivity.
* Every other combination is **PLAUSIBLE**, with the failed or unresolved leg
  named.  No majority vote and no post-hoc threshold decides.

## 9. Controls and provenance

The probe runs the regional audit's self-test and loader stamp refusals rather
than copying them.  Its new arithmetic has direct controls, each demonstrated
to fire on a planted violation:

1. exact per-row-to-band closure; a planted wrong row weight must fail;
2. thickness weighting; a planted layer average must disagree on a vertically
   sheared known field;
3. autocorrelation-aware `N_eff`; a planted independent-row denominator must
   fail on a duplicated-block profile;
4. persistence classifier; a planted row permutation must cross from the
   persistent to the decorrelating side;
5. cancellation classifier; a planted alternating profile must fire the
   cancellation/roughness bars while a same-signed smooth profile must not;
6. row-specific floors; replacing them with one global floor must change a
   known synthetic classification and be caught.

The run refuses stampless input through the imported loaders.  The output
records producer SHA, dirty state, input directories, horizons, channel rows,
all constants above, and the upstream regional-audit SHA.  It refuses a dirty
producer and a tree that changes during the run.
